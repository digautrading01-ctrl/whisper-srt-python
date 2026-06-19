# Whisper Offline Speech-to-Text → SRT (Flask)

A minimal Flask web app that:
- accepts audio uploads (`.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.webm`)
- transcribes **offline** using **faster-whisper** (CTranslate2-based Whisper)
- outputs an `.srt` subtitle file with timestamps
- shows a live transcription progress bar in the browser
- estimates time remaining during transcription
- lets you **pause/resume** an in-progress transcription from the UI to reduce resource usage
- lets you **cancel** a running (or paused) transcription from the UI
- prefers CUDA GPU acceleration automatically when available, and falls back to CPU otherwise

## 1) Prerequisites

### Python
Python 3.10+ recommended. Compatible with Python 3.14.

### ffmpeg (required)
This app uses `ffmpeg` to convert incoming audio into a Whisper-friendly format.

Install ffmpeg and make sure `ffmpeg` is available in your terminal `PATH`:
```bash
ffmpeg -version
```

### Whisper model (required)

Download a **CTranslate2-converted** Whisper model from the Hugging Face Hub:

https://huggingface.co/guillaumekln/faster-whisper-base/tree/main

Replace `base` in the URL with your preferred size (e.g. `faster-whisper-small`, `faster-whisper-medium`, `faster-whisper-large-v3`).

Then either:
- place the extracted model folder at: `model/`
  - the folder should contain `config.json` and `model.bin`

OR
- set an environment variable pointing at the model folder:
  - `WHISPER_MODEL_PATH=/path/to/faster-whisper-base`

If no local model is found, the app will fall back to downloading the model by name (e.g. `base`) from Hugging Face Hub on startup.

## 2) Install

From the project folder:
```bash
python -m venv .venv
```

Activate it:
- Windows (PowerShell):
```powershell
.\.venv\Scripts\Activate.ps1
```
- macOS/Linux:
```bash
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## 3) Run

```bash
python app.py
```

Open:
http://127.0.0.1:5000

## Notes / Customization

- Upload size limit defaults to `200MB`. You can change it via:
  - `MAX_CONTENT_LENGTH_MB=500`
- The SRT segmentation uses segment-level timestamps from Whisper for accurate timing.
- The web UI uploads the file, starts a background transcription job, then polls the server for progress.
- The transcription progress bar is based on processed Whisper segment timestamps compared with the converted WAV duration, so it reflects actual transcription progress instead of a generic spinner.
- The UI also estimates time remaining from observed transcription speed while the job is running.
- While a job is **paused**, the server stops pulling new segments from Whisper, which significantly reduces compute usage until you **resume**.
  - Pause is only available once the job is in the **Transcribing** stage (it does not pause `ffmpeg` conversion).
  - Cancelling is available during conversion, transcription, or while paused.
- Output writing is done via a temporary file + atomic rename to avoid corrupted `.srt` files.
- You can configure the Whisper model size (used as fallback when no local model is found) via environment variable:
  - `WHISPER_MODEL=base` (default, other options: tiny, tiny.en, base.en, small, small.en, medium, medium.en, large, large-v2, large-v3, distil-large-v2, distil-large-v3)
- By default the app uses `WHISPER_DEVICE=auto`, which tries CUDA first and automatically falls back to CPU if CUDA is unavailable or model initialization fails.
- You can still configure the runtime manually:
  - `WHISPER_DEVICE=auto` (default), `cpu`, or `cuda`
  - `WHISPER_COMPUTE_TYPE=auto` (default) for device-specific defaults, or set an explicit value such as `int8`, `float32`, or `float16`
- The `/healthz` endpoint now reports the active runtime device and compute type.
