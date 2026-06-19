import os
import subprocess
import threading
import time
import uuid
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from faster_whisper import WhisperModel
from flask import Flask, flash, jsonify, redirect, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

# Where you place the pre-downloaded CTranslate2 Whisper model folder.
# You can override with environment variable: WHISPER_MODEL_PATH
# If not set, falls back to downloading by model size name (e.g. "base").
MODEL_PATH = Path(os.environ.get("WHISPER_MODEL_PATH", str(BASE_DIR / "model")))

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}
MAX_CONTENT_LENGTH_MB = int(os.environ.get("MAX_CONTENT_LENGTH_MB", "200"))
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")
# Device: "auto" (default), "cpu", or "cuda".
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "auto").strip().lower()
# Compute type: leave unset/"auto" for sensible defaults per device, or set explicitly.
WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "auto").strip().lower()

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH_MB * 1024 * 1024


FINISHED_JOB_TTL_SECONDS = int(os.environ.get("FINISHED_JOB_TTL_SECONDS", "3600"))


@dataclass
class TranscriptionJob:
    job_id: str
    original_name: str
    status: str = "queued"
    stage: str = "Queued"
    detail: str = "Waiting to start."
    progress: float = 0.0
    error: str | None = None
    output_path: str | None = None
    download_name: str | None = None
    eta_seconds: int | None = None
    finished_at: float | None = None


def find_model_path() -> str:
    """
    Determine the model path to use.
    - If MODEL_PATH directory exists, use it as a local pre-downloaded model.
    - Otherwise, fall back to WHISPER_MODEL name (downloads from Hugging Face Hub).
    """
    if MODEL_PATH.is_dir():
        # Check it looks like a CTranslate2 model directory
        expected_files = {"config.json", "model.bin"}
        if any(f.name in expected_files for f in MODEL_PATH.iterdir()):
            print(f"Using local model directory: {MODEL_PATH}")
            return str(MODEL_PATH)
        # Directory exists but doesn't look like a model — warn and fall through
        print(f"Warning: {MODEL_PATH} exists but doesn't look like a CTranslate2 model. "
              f"Expected 'config.json' or 'model.bin' inside. Falling back to remote download.")

    print(f"No local model found. Will download '{WHISPER_MODEL}' from Hugging Face Hub.")
    return WHISPER_MODEL


def detect_nvidia_gpu() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        return result.returncode == 0 and "GPU" in result.stdout
    except Exception:
        return False


def default_compute_type_for(device: str) -> str:
    return "float16" if device == "cuda" else "int8"


def normalize_device(value: str) -> str:
    return value if value in {"auto", "cpu", "cuda"} else "auto"


def resolve_device_attempts() -> list[tuple[str, str]]:
    requested_device = normalize_device(WHISPER_DEVICE)
    requested_compute = "" if WHISPER_COMPUTE_TYPE in {"", "auto"} else WHISPER_COMPUTE_TYPE
    attempts: list[tuple[str, str]] = []

    def add_attempt(device: str, compute_type: str | None = None) -> None:
        attempt = (device, compute_type or default_compute_type_for(device))
        if attempt not in attempts:
            attempts.append(attempt)

    if requested_device == "cpu":
        add_attempt("cpu", requested_compute)
        return attempts

    if requested_device == "cuda":
        add_attempt("cuda", requested_compute or "float16")
        add_attempt("cpu", "int8")
        return attempts

    if detect_nvidia_gpu():
        add_attempt("cuda", requested_compute or "float16")
    add_attempt("cpu", "int8")
    return attempts


def load_whisper_model() -> tuple[WhisperModel, str, str]:
    model_path = find_model_path()
    use_local = Path(model_path).is_dir()
    errors: list[str] = []

    for device, compute_type in resolve_device_attempts():
        try:
            print(f"Loading Whisper model: {model_path} (device={device}, compute_type={compute_type})...")
            loaded_model = WhisperModel(
                model_path,
                device=device,
                compute_type=compute_type,
                local_files_only=use_local,
            )
            print(f"Whisper model loaded on {device} with compute type {compute_type}.")
            return loaded_model, device, compute_type
        except Exception as exc:
            errors.append(f"{device}/{compute_type}: {exc}")
            print(f"Unable to load Whisper model on {device} with compute type {compute_type}: {exc}")

    joined_errors = "\n".join(errors) if errors else "No device attempts were made."
    raise RuntimeError(f"Failed to load Whisper model.\n{joined_errors}")


# Load Whisper model once at startup
model, MODEL_RUNTIME_DEVICE, MODEL_RUNTIME_COMPUTE_TYPE = load_whisper_model()


JOBS: dict[str, TranscriptionJob] = {}
JOBS_LOCK = threading.Lock()

class JobCancelledError(RuntimeError):
    pass


@dataclass
class JobControl:
    pause_event: threading.Event
    cancel_event: threading.Event
    lock: threading.Lock
    resume_stage: str | None = None
    resume_detail: str | None = None


JOB_CONTROLS: dict[str, JobControl] = {}
JOB_CONTROLS_LOCK = threading.Lock()


def create_job_control(job_id: str) -> JobControl:
    control = JobControl(
        pause_event=threading.Event(),
        cancel_event=threading.Event(),
        lock=threading.Lock(),
    )
    control.pause_event.set()  # running by default
    with JOB_CONTROLS_LOCK:
        JOB_CONTROLS[job_id] = control
    return control


def get_job_control(job_id: str) -> JobControl | None:
    with JOB_CONTROLS_LOCK:
        return JOB_CONTROLS.get(job_id)


def remove_job_control(job_id: str) -> None:
    with JOB_CONTROLS_LOCK:
        JOB_CONTROLS.pop(job_id, None)


def wait_if_paused(control: JobControl | None) -> None:
    if not control:
        return
    while not control.pause_event.is_set():
        if control.cancel_event.is_set():
            raise JobCancelledError("Cancelled by user.")
        control.pause_event.wait(timeout=0.25)
    if control.cancel_event.is_set():
        raise JobCancelledError("Cancelled by user.")


def check_cancelled(control: JobControl | None) -> None:
    if control and control.cancel_event.is_set():
        raise JobCancelledError("Cancelled by user.")


@dataclass
class SrtSegment:
    index: int
    start: float
    end: float
    text: str


def ensure_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_job(job: TranscriptionJob) -> None:
    with JOBS_LOCK:
        JOBS[job.job_id] = job


def get_job(job_id: str) -> TranscriptionJob | None:
    with JOBS_LOCK:
        return JOBS.get(job_id)


_TERMINAL_STATUSES = {"complete", "error", "cancelled"}


def _cleanup_finished_jobs() -> None:
    """Remove terminal jobs that finished more than FINISHED_JOB_TTL_SECONDS ago."""
    cutoff = time.time() - FINISHED_JOB_TTL_SECONDS
    with JOBS_LOCK:
        to_remove = [
            jid for jid, job in JOBS.items()
            if job.finished_at is not None and job.finished_at < cutoff
        ]
        for jid in to_remove:
            del JOBS[jid]


def job_payload(job: TranscriptionJob) -> dict:
    payload = asdict(job)
    payload.pop("finished_at", None)
    payload["progress"] = round(job.progress, 1)
    payload["pause_url"] = url_for("pause_job", job_id=job.job_id)
    payload["resume_url"] = url_for("resume_job", job_id=job.job_id)
    payload["cancel_url"] = url_for("cancel_job", job_id=job.job_id)
    if job.status == "complete":
        payload["download_url"] = url_for("download_job_output", job_id=job.job_id)
    return payload


def update_job(
    job_id: str,
    *,
    status: str | None = None,
    stage: str | None = None,
    detail: str | None = None,
    progress: float | None = None,
    error: str | None = None,
    output_path: str | None = None,
    download_name: str | None = None,
    eta_seconds: int | None = None,
) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        if status is not None:
            job.status = status
            if status in _TERMINAL_STATUSES and job.finished_at is None:
                job.finished_at = time.time()
        if stage is not None:
            job.stage = stage
        if detail is not None:
            job.detail = detail
        if progress is not None:
            job.progress = max(0.0, min(100.0, progress))
        if error is not None:
            job.error = error
        if output_path is not None:
            job.output_path = output_path
        if download_name is not None:
            job.download_name = download_name
        job.eta_seconds = eta_seconds


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def is_ajax_request() -> bool:
    return request.headers.get("X-Requested-With") == "XMLHttpRequest"


def error_out(message: str, status_code: int = 400):
    """
    Return an error in a way that works for both:
      - normal form POST (flash + redirect)
      - AJAX upload (JSON response)
    """
    if is_ajax_request():
        return jsonify({"error": message}), status_code
    flash(message, "error")
    return redirect(url_for("index"))


def run_ffmpeg_convert(src_path: Path, dst_path: Path, *, control: JobControl | None = None) -> None:
    """
    Convert audio into a format suitable for Whisper using ffmpeg.
    Outputs WAV 16kHz mono 16-bit PCM.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(dst_path),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        while True:
            if control and control.cancel_event.is_set():
                try:
                    proc.terminate()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=3)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                raise JobCancelledError("Cancelled by user.")

            rc = proc.poll()
            if rc is not None:
                break
            time.sleep(0.2)

        stdout, stderr = proc.communicate(timeout=1)
    except subprocess.TimeoutExpired:
        stdout, stderr = "", ""

    if proc.returncode != 0:
        raise RuntimeError(
            "ffmpeg conversion failed. Ensure ffmpeg is installed and available in PATH.\n"
            f"stderr: {(stderr or '').strip()}"
        )


def get_wav_duration_seconds(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as wav_file:
        frame_rate = wav_file.getframerate()
        if frame_rate <= 0:
            raise RuntimeError("Unable to determine WAV frame rate.")
        return wav_file.getnframes() / frame_rate


def format_srt_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    ms = int(round((seconds - int(seconds)) * 1000))
    total_seconds = int(seconds)
    s = total_seconds % 60
    m = (total_seconds // 60) % 60
    h = total_seconds // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def split_into_two_lines(text: str, max_line_len: int = 42) -> str:
    """
    Basic SRT line wrapping: return 1-2 lines. Keeps it simple and predictable.
    """
    text = " ".join(text.split())
    if len(text) <= max_line_len:
        return text
    words = text.split()
    mid = max(1, len(words) // 2)
    line1 = " ".join(words[:mid])
    line2 = " ".join(words[mid:])
    # If still too long, fall back to single line (player will wrap).
    if len(line1) > max_line_len * 1.4 or len(line2) > max_line_len * 1.4:
        return text
    return line1 + "\n" + line2


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def format_progress_detail(processed_seconds: float, total_duration: float, eta_seconds: int | None) -> str:
    detail = f"Processed {format_duration(processed_seconds)} of {format_duration(total_duration)}."
    if eta_seconds is not None:
        detail += f" About {format_duration(eta_seconds)} remaining."
    else:
        detail += " Estimating time remaining."
    return detail


def estimate_eta_seconds(progress_pct: float, elapsed_seconds: float) -> int | None:
    if progress_pct <= 0 or elapsed_seconds <= 0:
        return None
    remaining = elapsed_seconds * ((100.0 - progress_pct) / progress_pct)
    if remaining < 0:
        return 0
    return int(round(remaining))


def transcribe_audio(
    audio_path: Path,
    *,
    total_duration: float | None = None,
    progress_callback: Callable[[float, float], None] | None = None,
    control: JobControl | None = None,
) -> list[SrtSegment]:
    """
    Transcribe audio using faster-whisper and return SRT segments.
    """
    segments_iter, info = model.transcribe(
        str(audio_path),
        beam_size=5,
        word_timestamps=False,
    )
    print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")

    segments = []
    max_progress = 0.0
    for seg in segments_iter:
        wait_if_paused(control)
        text = seg.text.strip()
        if text:
            text = split_into_two_lines(text)
            segments.append(SrtSegment(
                index=len(segments) + 1,
                start=round(seg.start, 3),
                end=round(seg.end, 3),
                text=text
            ))
        if progress_callback and total_duration and total_duration > 0:
            segment_end = max(0.0, min(float(seg.end), total_duration))
            progress_pct = min(99.0, (segment_end / total_duration) * 100.0)
            if progress_pct >= max_progress:
                max_progress = progress_pct
                progress_callback(
                    progress_pct,
                    segment_end,
                )
    return segments


def segments_to_srt(segments: list[SrtSegment]) -> str:
    lines: list[str] = []
    for seg in segments:
        start_ts = format_srt_timestamp(seg.start)
        end_ts = format_srt_timestamp(seg.end)
        lines.append(str(seg.index))
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(seg.text)
        lines.append("")  # blank line
    return "\n".join(lines).strip() + "\n"


def run_transcription_job(
    job_id: str,
    *,
    src_path: Path,
    wav_path: Path,
    srt_path: Path,
    download_name: str,
) -> None:
    control = get_job_control(job_id)
    try:
        check_cancelled(control)
        update_job(job_id, status="running", stage="Preparing Audio", detail="Converting upload to 16 kHz WAV.", progress=2.0)
        run_ffmpeg_convert(src_path, wav_path, control=control)

        wait_if_paused(control)
        total_duration = get_wav_duration_seconds(wav_path)
        update_job(
            job_id,
            stage="Transcribing",
            detail=format_progress_detail(0.0, total_duration, None),
            progress=3.0,
            eta_seconds=None,
        )
        transcription_started_at = time.perf_counter()

        def on_progress(progress_pct: float, processed_seconds: float) -> None:
            # If the user paused/cancelled, don't compute ETA while blocked.
            wait_if_paused(control)
            elapsed_seconds = time.perf_counter() - transcription_started_at
            eta_seconds = estimate_eta_seconds(progress_pct, elapsed_seconds)
            detail = format_progress_detail(processed_seconds, total_duration, eta_seconds)
            update_job(
                job_id,
                stage="Transcribing",
                detail=detail,
                progress=progress_pct,
                eta_seconds=eta_seconds,
            )

        segments = transcribe_audio(
            wav_path,
            total_duration=total_duration,
            progress_callback=on_progress,
            control=control,
        )
        if not segments:
            raise RuntimeError("No speech detected (or transcription returned empty).")

        update_job(job_id, stage="Writing Output", detail="Saving subtitle file.", progress=99.0, eta_seconds=0)
        srt_text = segments_to_srt(segments)
        tmp_path = srt_path.with_suffix(srt_path.suffix + ".tmp")
        tmp_path.write_text(srt_text, encoding="utf-8")
        tmp_path.replace(srt_path)
        update_job(
            job_id,
            status="complete",
            stage="Done",
            detail=f"Finished. Ready to download {download_name}.",
            progress=100.0,
            output_path=str(srt_path),
            download_name=download_name,
            eta_seconds=0,
        )
    except JobCancelledError:
        update_job(
            job_id,
            status="cancelled",
            stage="Cancelled",
            detail="Cancelled by user.",
            error=None,
            eta_seconds=None,
        )
    except Exception as exc:
        update_job(
            job_id,
            status="error",
            stage="Failed",
            detail="The transcription job failed.",
            error=str(exc),
            eta_seconds=None,
        )
    finally:
        # Cleanup temp output if present (avoid corrupted partial files).
        try:
            tmp_path = srt_path.with_suffix(srt_path.suffix + ".tmp")
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        job = get_job(job_id)
        if job and job.status != "complete":
            try:
                if srt_path.exists():
                    srt_path.unlink()
            except Exception:
                pass
        for p in (src_path, wav_path):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass
        remove_job_control(job_id)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", runtime_device=MODEL_RUNTIME_DEVICE.upper())


@app.route("/transcribe", methods=["POST"])
def transcribe():
    ensure_dirs()
    _cleanup_finished_jobs()

    if "audio" not in request.files:
        return error_out("No file uploaded.")

    f = request.files["audio"]
    if not f.filename:
        return error_out("Please choose a file.")

    if not allowed_file(f.filename):
        return error_out("Unsupported format. Please upload a supported audio file.")

    job_id = uuid.uuid4().hex
    original_name = secure_filename(f.filename)
    src_ext = Path(original_name).suffix.lower()

    src_path = UPLOAD_DIR / f"{job_id}{src_ext}"
    wav_path = UPLOAD_DIR / f"{job_id}_converted.wav"
    download_name = f"{Path(original_name).stem}.srt"
    srt_path = OUTPUT_DIR / f"{job_id}_{download_name}"

    try:
        f.save(src_path)
        job = TranscriptionJob(job_id=job_id, original_name=original_name)
        save_job(job)
        create_job_control(job_id)
        worker = threading.Thread(
            target=run_transcription_job,
            kwargs={
                "job_id": job_id,
                "src_path": src_path,
                "wav_path": wav_path,
                "srt_path": srt_path,
                "download_name": download_name,
            },
            daemon=True,
        )
        worker.start()
        if is_ajax_request():
            return jsonify({
                "job_id": job_id,
                "status_url": url_for("job_status", job_id=job_id),
                "download_url": url_for("download_job_output", job_id=job_id),
                "pause_url": url_for("pause_job", job_id=job_id),
                "resume_url": url_for("resume_job", job_id=job_id),
                "cancel_url": url_for("cancel_job", job_id=job_id),
            }), 202
        flash("Transcription started. This page supports live progress when JavaScript is enabled.", "message")
        return redirect(url_for("index"))
    except Exception as e:
        return error_out(str(e))


@app.route("/jobs/<job_id>", methods=["GET"])
def job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found."}), 404
    return jsonify(job_payload(job))


def _job_control_error(message: str, status_code: int = 409):
    return jsonify({"error": message}), status_code


@app.route("/jobs/<job_id>/pause", methods=["POST"])
def pause_job(job_id: str):
    job = get_job(job_id)
    if not job:
        return _job_control_error("Job not found.", 404)
    control = get_job_control(job_id)
    if not control:
        return _job_control_error("Job control not available.", 409)
    if job.status != "running" or job.stage != "Transcribing":
        return _job_control_error("Job can only be paused while transcribing.", 409)
    with control.lock:
        # Snapshot current stage/detail so we can restore them on resume.
        control.resume_stage = job.stage
        control.resume_detail = job.detail
        control.pause_event.clear()
        update_job(
            job_id,
            status="paused",
            stage="Paused",
            detail="Paused by user. Click Resume to continue.",
            eta_seconds=None,
        )
    return jsonify(job_payload(get_job(job_id)))


@app.route("/jobs/<job_id>/resume", methods=["POST"])
def resume_job(job_id: str):
    job = get_job(job_id)
    if not job:
        return _job_control_error("Job not found.", 404)
    control = get_job_control(job_id)
    if not control:
        return _job_control_error("Job control not available.", 409)
    if job.status != "paused":
        return _job_control_error("Job is not paused.", 409)
    with control.lock:
        control.pause_event.set()
        update_job(
            job_id,
            status="running",
            stage=control.resume_stage or "Transcribing",
            detail=control.resume_detail or "Resuming transcription…",
            eta_seconds=None,
        )
    return jsonify(job_payload(get_job(job_id)))


@app.route("/jobs/<job_id>/cancel", methods=["POST"])
def cancel_job(job_id: str):
    job = get_job(job_id)
    if not job:
        return _job_control_error("Job not found.", 404)
    control = get_job_control(job_id)
    if not control:
        return _job_control_error("Job control not available.", 409)
    if job.status in {"complete", "error", "cancelled"}:
        return jsonify(job_payload(job))
    with control.lock:
        control.cancel_event.set()
        control.pause_event.set()  # unblock any paused wait
        update_job(
            job_id,
            status="cancelling",
            stage="Cancelling",
            detail="Cancelling…",
            eta_seconds=None,
        )
    return jsonify(job_payload(get_job(job_id)))

@app.route("/jobs/<job_id>/download", methods=["GET"])
def download_job_output(job_id: str):
    job = get_job(job_id)
    if not job:
        return error_out("Job not found.", 404)
    if job.status != "complete" or not job.output_path or not job.download_name:
        return error_out("Output is not ready yet.", 409)
    return send_file(
        job.output_path,
        as_attachment=True,
        download_name=job.download_name,
        mimetype="application/x-subrip",
    )


@app.route("/healthz", methods=["GET"])
def healthz():
    return {
        "ok": True,
        "device": MODEL_RUNTIME_DEVICE,
        "compute_type": MODEL_RUNTIME_COMPUTE_TYPE,
    }


if __name__ == "__main__":
    ensure_dirs()
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=True)
