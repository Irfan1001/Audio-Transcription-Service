"""Whisper transcription pipeline: normalize, chunk, transcribe, merge timestamps."""

from __future__ import annotations

import asyncio
import logging
import tempfile
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from app.core.config import Settings, get_settings
from app.core.errors import ModelNotLoadedError, TranscriptionError
from app.models.schemas import TranscriptionResponse, TranscriptSegment
from app.services import audio

logger = logging.getLogger(__name__)


class WhisperSegment(Protocol):
    start: float
    end: float
    text: str


class WhisperInfo(Protocol):
    language: str
    language_probability: float


class WhisperModelLike(Protocol):
    """The small part of faster-whisper's WhisperModel that this service uses."""

    def transcribe(self, audio: str) -> tuple[Iterable[WhisperSegment], WhisperInfo]: ...


@dataclass(frozen=True)
class ChunkResult:
    start_offset_seconds: float
    language: str
    language_probability: float | None
    segments: list[TranscriptSegment]


def offset_segments(segments: Sequence[TranscriptSegment], offset_seconds: float) -> list[TranscriptSegment]:
    """Shift chunk-relative timestamps to be relative to the original audio."""
    return [
        TranscriptSegment(
            start=round(segment.start + offset_seconds, 3),
            end=round(segment.end + offset_seconds, 3),
            text=segment.text,
        )
        for segment in segments
    ]


def merge_chunk_results(results: Sequence[ChunkResult]) -> list[TranscriptSegment]:
    """Apply each chunk's offset and return all segments in chronological order."""
    merged: list[TranscriptSegment] = []
    for result in results:
        merged.extend(offset_segments(result.segments, result.start_offset_seconds))
    merged.sort(key=lambda segment: (segment.start, segment.end))
    return merged


def detect_language(results: Sequence[ChunkResult]) -> tuple[str, float | None]:
    """Pick the language of the chunk Whisper was most confident about."""
    if not results:
        return "unknown", None
    best = max(results, key=lambda result: result.language_probability or 0.0)
    return best.language, best.language_probability


class TranscriptionService:
    """Loads the Whisper model once and runs the transcription pipeline."""

    def __init__(self, settings: Settings | None = None, model: WhisperModelLike | None = None) -> None:
        self._settings = settings or get_settings()
        self._model = model
        self._slots = asyncio.Semaphore(max(1, self._settings.max_concurrent_transcriptions))

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    def load_model(self) -> None:
        from faster_whisper import WhisperModel

        settings = self._settings
        logger.info(
            "Loading Whisper model '%s' (device=%s, compute_type=%s)",
            settings.whisper_model, settings.whisper_device, settings.whisper_compute_type,
        )
        self._model = WhisperModel(
            settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )
        logger.info("Whisper model loaded")

    async def transcribe(self, source: Path) -> TranscriptionResponse:
        """Run the pipeline in a worker thread so the event loop stays responsive."""
        async with self._slots:
            return await asyncio.to_thread(self.transcribe_sync, source)

    def transcribe_sync(self, source: Path) -> TranscriptionResponse:
        audio.ensure_ffmpeg_available()

        with tempfile.TemporaryDirectory(prefix="transcription-") as workdir:
            work = Path(workdir)
            normalized = audio.normalize_audio(source, work / "normalized.wav")
            duration = audio.get_duration_seconds(normalized)
            chunks = audio.split_into_chunks(normalized, self._settings.chunk_duration_seconds, work)

            logger.info("Transcription started: duration=%.2fs chunks=%d", duration, len(chunks))
            results = [self._transcribe_chunk_with_retry(chunk) for chunk in chunks]

        language, probability = detect_language(results)
        segments = merge_chunk_results(results)
        logger.info("Transcription completed: language=%s segments=%d", language, len(segments))

        return TranscriptionResponse(
            language=language,
            language_probability=probability,
            duration_seconds=round(duration, 3),
            chunk_count=len(chunks),
            segments=segments,
        )

    def _transcribe_chunk_with_retry(self, chunk: audio.AudioChunk) -> ChunkResult:
        attempts = max(1, self._settings.max_transcription_retries)
        for attempt in range(1, attempts + 1):
            try:
                return self._transcribe_chunk(chunk)
            except ModelNotLoadedError:
                raise
            except Exception as exc:
                if attempt == attempts:
                    raise TranscriptionError(
                        f"Transcription of '{chunk.path.name}' failed after {attempts} attempt(s): {exc}"
                    ) from exc
                logger.warning(
                    "Retry %d/%d for chunk '%s' after error: %s", attempt, attempts, chunk.path.name, exc
                )
                time.sleep(self._settings.retry_delay_seconds * attempt)
        raise TranscriptionError("Unreachable: retry loop exited without a result.")

    def _transcribe_chunk(self, chunk: audio.AudioChunk) -> ChunkResult:
        if self._model is None:
            raise ModelNotLoadedError("Whisper model is not loaded.")

        raw_segments, info = self._model.transcribe(str(chunk.path))
        return ChunkResult(
            start_offset_seconds=chunk.start_offset_seconds,
            language=info.language,
            language_probability=getattr(info, "language_probability", None),
            segments=self._clean_segments(raw_segments, chunk.duration_seconds),
        )

    @staticmethod
    def _clean_segments(raw_segments: Iterable[WhisperSegment], chunk_duration: float) -> list[TranscriptSegment]:
        """Drop empty segments and keep timestamps inside the chunk Whisper was given."""
        segments: list[TranscriptSegment] = []
        for raw in raw_segments:
            text = raw.text.strip()
            start = min(max(raw.start, 0.0), chunk_duration)
            end = min(max(raw.end, start), chunk_duration)
            if text:
                segments.append(TranscriptSegment(start=round(start, 3), end=round(end, 3), text=text))
        return segments


@lru_cache
def get_transcription_service() -> TranscriptionService:
    """Return the process-wide service (also used as a FastAPI dependency)."""
    return TranscriptionService()
