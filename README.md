# Audio Transcription Service

A small FastAPI service that accepts an audio file, normalizes it with FFmpeg, transcribes it with
an open-source Whisper model 

## Overview

- Accepts WAV, MP3, M4A, FLAC and OGG uploads.
- Normalizes every input to 16 kHz mono 16-bit PCM WAV with FFmpeg, so the model always sees the same format.
- Splits long audio into fixed-length chunks and transcribes them one by one.
- Adds each chunk's start offset back to the timestamps, so the response is relative to the original recording.
- Retries a failed chunk a few times before giving up.
- Loads the Whisper model once at startup and reuses it for every request.

## Architecture

```text
Client
  |
  | POST /transcribe
  v
FastAPI            (validate extension, stream upload to a temp file, enforce size limit)
  |
  v
FFmpeg             (normalize -> WAV / PCM s16 / 16 kHz / mono)
  |
  v
Audio Chunks       (CHUNK_DURATION_SECONDS, only when the audio is longer than one chunk)
  |
  v
Whisper large-v3   (one call per chunk, with retries)
  |
  v
Timestamp Merge    (segment.start + chunk offset, sorted chronologically)
  |
  v
JSON Response
```

### Project layout

```text
app/
    main.py                     FastAPI app, lifespan (model load), error handler
    api/routes.py               HTTP endpoints (thin; no business logic)
    services/audio.py           validation, upload handling, FFmpeg normalization, chunking
    services/transcription.py   Whisper model, retries, timestamp merging
    models/schemas.py           Pydantic request/response models
    core/config.py              settings from environment variables
    core/errors.py              domain errors with their HTTP status codes
    core/logging_config.py      logging setup
tests/                          pytest suite (Whisper is mocked)
```

## Local setup

```bash
git clone <repo>
cd <repo>

python3.11 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt        # runtime only
pip install -r requirements-dev.txt    # runtime + pytest
```

Python 3.11+ is required.

### Install FFmpeg

FFmpeg (and `ffprobe`, which ships with it) must be on your `PATH`.

| Platform | Command |
| --- | --- |
| macOS | `brew install ffmpeg` |
| Ubuntu/Debian | `sudo apt update && sudo apt install ffmpeg` |
| Windows | `winget install Gyan.FFmpeg` (or `choco install ffmpeg`) |

Verify with `ffmpeg -version`. If FFmpeg is missing, startup logs a warning and `/transcribe`
returns `503` with a clear message.

### Run the service

```bash
uvicorn app.main:app --reload
```

On first start the Whisper model is downloaded from Hugging Face and cached in
`~/.cache/huggingface`. `large-v3` is roughly 3 GB, so the first startup takes a while. For a quick
local check, use a smaller model:

```bash
WHISPER_MODEL=tiny WHISPER_COMPUTE_TYPE=int8 uvicorn app.main:app --reload
```

Interactive API docs: <http://localhost:8000/docs>

### Try it

```bash
curl -X POST \
  -F "file=@sample.mp3" \
  http://localhost:8000/transcribe
```

  

### Run the tests

```bash
pytest
```

The tests mock Whisper, so nothing is downloaded. Tests that exercise FFmpeg are skipped
automatically if FFmpeg is not installed.

## API

### `POST /transcribe`

Multipart form upload, field name `file`.

```json
{
  "language": "en",
  "language_probability": 0.98,
  "duration_seconds": 65.8,
  "chunk_count": 3,
  "segments": [
    { "start": 0.52, "end": 3.41, "text": "Hello, how are you?" },
    { "start": 3.62, "end": 7.85, "text": "I'm calling regarding my account." }
  ]
}
```

Errors are returned as `{"detail": "..."}`:

| Status | Cause |
| --- | --- |
| 400 | Empty upload |
| 413 | File larger than `MAX_FILE_SIZE_MB` |
| 415 | Unsupported file extension |
| 422 | Missing `file` field, or FFmpeg could not decode the audio |
| 500 | Transcription failed after all retries |
| 503 | FFmpeg not installed, or the model is not loaded |

### `GET /health`

```json
{
  "status": "ok",
  "whisper_model": "large-v3",
  "model_loaded": true,
  "ffmpeg_available": true
}
```

### `GET /`

Service name, version and the list of supported extensions.

## Configuration

All values are read from environment variables (or a `.env` file — see `.env.example`) and have
working defaults, so the service runs without any configuration.

| Variable | Default | Purpose |
| --- | --- | --- |
| `WHISPER_MODEL` | `large-v3` | faster-whisper model name or local path |
| `WHISPER_DEVICE` | `auto` | `auto`, `cpu` or `cuda` |
| `WHISPER_COMPUTE_TYPE` | `default` | e.g. `int8` on CPU, `float16` on GPU |
| `PRELOAD_MODEL` | `true` | Set to `false` to start without loading the model (used by tests) |
| `CHUNK_DURATION_SECONDS` | `30` | Chunk length for long audio |
| `MAX_FILE_SIZE_MB` | `100` | Upload size limit |
| `MAX_TRANSCRIPTION_RETRIES` | `3` | Attempts per chunk |
| `RETRY_DELAY_SECONDS` | `1.0` | Base delay between retries (multiplied by the attempt number) |
| `MAX_CONCURRENT_TRANSCRIPTIONS` | `1` | How many transcriptions may run at once |
| `LOG_LEVEL` | `INFO` | Python logging level |

## Design decisions

### Why FastAPI?

It gives a typed HTTP API with multipart upload support, automatic validation from the Pydantic
models, and generated OpenAPI docs, with very little code.

### Why Whisper (faster-whisper)?

Whisper is an open-source speech-to-text model, so there are no per-minute API costs and no audio
leaves the machine. `faster-whisper` is a CTranslate2 reimplementation that is significantly faster
and lighter on memory than the reference implementation, and it returns per-segment timestamps
directly.

### Why FFmpeg?

FFmpeg decodes essentially every audio container and codec, which is what makes "accepts WAV or
MP3 or M4A or FLAC or OGG" a few CLI flags instead of a format-handling layer. It is also the
practical way to cut audio into chunks.

### Why 16 kHz mono?

Whisper is trained on 16 kHz mono audio, and speech carries almost no useful information above
8 kHz. Downmixing and resampling gives the model exactly the format it expects and shrinks the data
it has to process. The uploaded file is never assumed to already be in that format ,  it is always
run through FFmpeg.

### Why chunk long audio?

Long recordings are transcribed in bounded pieces, which keeps memory use flat and gives useful
progress in the logs. Chunking is skipped entirely when the audio is shorter than one chunk, so
short files pay no extra cost.

### How are timestamps handled?

Whisper reports timestamps relative to whatever audio it was given, so a chunk starting at 60 s
returns `start=2.5` for something that happens at 62.5 s. Each chunk records its own
`start_offset_seconds`, and `merge_chunk_results()` adds that offset to every segment and sorts the
result chronologically. This is the piece of logic that the tests cover most directly.

No overlap between chunks is used. Overlap avoids clipping a word at a boundary but then requires
de-duplicating text across chunks, which is exactly the kind of complexity this assignment asks to
avoid.

### How is concurrency handled?

The endpoints are `async`, so FastAPI can accept many uploads at once. Whisper inference, FFmpeg and
file I/O are all blocking and CPU-bound, so the pipeline runs in a worker thread via
`asyncio.to_thread` — that keeps the event loop free to accept and validate other requests. An
`asyncio.Semaphore` (`MAX_CONCURRENT_TRANSCRIPTIONS`, default 1) then limits how many transcriptions
actually run at the same time, because running several large-v3 inferences in one process would only
make them all slower and risk exhausting memory. Async does not make Whisper inference parallel; it
only stops one slow request from blocking the whole server.

### Storage

Uploads are streamed to a `tempfile.TemporaryDirectory` and deleted when the request finishes;
normalized audio and chunks live in a second temporary directory that is removed the same way.
Nothing is persisted, and the transcription is returned directly in the HTTP response.

### How would this scale?

```text
                    API
                     |
                     v
                   Queue
             ┌───────┼───────┐
             v       v       v
          Worker  Worker  Worker
             |       |       |
          Whisper Whisper Whisper
             └───────┼───────┘
                     v
                  Storage
```

In production the request would not block on inference. `POST /transcribe` would upload the audio to
object storage, enqueue a job and return a job id; a pool of GPU workers (one Whisper model each)
would consume the queue, and the client would poll or receive a webhook. Roughly:

```text
Audio      -> S3/object storage
Metadata   -> PostgreSQL
Jobs       -> Queue
Workers    -> Whisper workers
```

Chunks are independent, so a long file could also be fanned out across workers and reassembled by
offset — the merging logic here is already the same operation.

This implementation intentionally stays a single local process with no queue, no database and no
object storage, because that is what the assignment asks for and it keeps the pipeline easy to read
end to end.

 
