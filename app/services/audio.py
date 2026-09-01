"""Audio validation, FFmpeg normalization and chunking."""

from __future__ import annotations

import logging
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from fastapi import UploadFile

from app.core.errors import (
    AudioProcessingError,
    EmptyAudioError,
    FFmpegNotFoundError,
    FileTooLargeError,
    UnsupportedAudioFormatError,
)

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: tuple[str, ...] = (".wav", ".mp3", ".m4a", ".flac", ".ogg")
TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1
TARGET_CODEC = "pcm_s16le"

_UPLOAD_READ_SIZE = 1024 * 1024
_MIN_CHUNK_SECONDS = 0.1


@dataclass(frozen=True)
class AudioChunk:
    """One chunk of normalized audio plus its offset in the original recording."""

    path: Path
    start_offset_seconds: float
    duration_seconds: float


def ffmpeg_available() -> bool:
    return all(shutil.which(tool) is not None for tool in ("ffmpeg", "ffprobe"))


def ensure_ffmpeg_available() -> None:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise FFmpegNotFoundError(
            f"Required tool(s) not found on PATH: {', '.join(missing)}. Install FFmpeg to use this service."
        )


def validate_extension(filename: str | None) -> str:
    """Return the lowercase extension of an allowed audio file, or raise."""
    suffix = Path(filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedAudioFormatError(
            f"Unsupported file type '{suffix or filename or 'unknown'}'. "
            f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}."
        )
    return suffix


async def save_upload(file: UploadFile, directory: Path, max_bytes: int) -> Path:
    """Stream an upload to disk, enforcing the extension and size limits."""
    suffix = validate_extension(file.filename)
    destination = directory / f"upload{suffix}"
    size = 0

    with destination.open("wb") as target:
        while data := await file.read(_UPLOAD_READ_SIZE):
            size += len(data)
            if size > max_bytes:
                raise FileTooLargeError(f"File exceeds the maximum size of {max_bytes // (1024 * 1024)} MB.")
            target.write(data)

    if size == 0:
        raise EmptyAudioError("Uploaded file is empty.")

    logger.info("File received: name=%s bytes=%d", file.filename, size)
    return destination


def normalize_audio(source: Path, destination: Path) -> Path:
    """Convert any supported input to 16 kHz mono 16-bit PCM WAV."""
    _run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source),
            "-vn", "-map", "0:a:0",
            "-ac", str(TARGET_CHANNELS),
            "-ar", str(TARGET_SAMPLE_RATE),
            "-acodec", TARGET_CODEC,
            str(destination),
        ]
    )
    logger.info("File normalized: %s -> %s (%d Hz, mono)", source.name, destination.name, TARGET_SAMPLE_RATE)
    return destination


def get_duration_seconds(path: Path) -> float:
    result = _run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise AudioProcessingError(f"Could not read the duration of '{path.name}'.") from exc


def split_into_chunks(path: Path, chunk_duration_seconds: int, output_dir: Path) -> list[AudioChunk]:
    """Split normalized audio into fixed-length chunks, keeping each chunk's offset."""
    if chunk_duration_seconds <= 0:
        raise AudioProcessingError("CHUNK_DURATION_SECONDS must be greater than zero.")

    duration = get_duration_seconds(path)
    if duration <= chunk_duration_seconds:
        return [AudioChunk(path=path, start_offset_seconds=0.0, duration_seconds=duration)]

    chunks: list[AudioChunk] = []
    for index in range(math.ceil(duration / chunk_duration_seconds)):
        offset = float(index * chunk_duration_seconds)
        chunk_path = output_dir / f"chunk_{index:04d}.wav"
        _run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", str(offset), "-t", str(chunk_duration_seconds),
                "-i", str(path),
                "-acodec", TARGET_CODEC,
                str(chunk_path),
            ]
        )
        chunk_duration = get_duration_seconds(chunk_path)
        if chunk_duration < _MIN_CHUNK_SECONDS:
            chunk_path.unlink(missing_ok=True)
            continue
        chunks.append(
            AudioChunk(path=chunk_path, start_offset_seconds=offset, duration_seconds=chunk_duration)
        )

    logger.info("Audio split into %d chunk(s) of up to %ds", len(chunks), chunk_duration_seconds)
    return chunks


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:
        raise FFmpegNotFoundError(f"'{command[0]}' not found on PATH. Install FFmpeg to use this service.") from exc
    except subprocess.CalledProcessError as exc:
        raise AudioProcessingError(f"{command[0]} failed: {_last_error_line(exc.stderr)}") from exc


def _last_error_line(stderr: str | None) -> str:
    lines = [line.strip() for line in (stderr or "").splitlines() if line.strip()]
    return lines[-1] if lines else "no error output"
