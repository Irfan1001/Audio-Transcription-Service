from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import Settings
from app.core.errors import TranscriptionError
from app.models.schemas import TranscriptSegment
from app.services.audio import AudioChunk
from app.services.transcription import (
    ChunkResult,
    TranscriptionService,
    detect_language,
    merge_chunk_results,
    offset_segments,
)
from tests.conftest import FakeSegment, FakeWhisperModel, requires_ffmpeg


def test_offset_segments_adds_the_chunk_start() -> None:
    segments = [TranscriptSegment(start=2.5, end=5.8, text="Hello")]

    shifted = offset_segments(segments, offset_seconds=60.0)

    assert shifted == [TranscriptSegment(start=62.5, end=65.8, text="Hello")]


def test_merge_chunk_results_applies_offsets_in_chronological_order() -> None:
    results = [
        ChunkResult(0.0, "en", 0.9, [TranscriptSegment(start=0.5, end=2.0, text="Hello")]),
        ChunkResult(30.0, "en", 0.9, [TranscriptSegment(start=1.0, end=3.0, text="How are you?")]),
    ]

    merged = merge_chunk_results(results)

    assert [(segment.start, segment.end, segment.text) for segment in merged] == [
        (0.5, 2.0, "Hello"),
        (31.0, 33.0, "How are you?"),
    ]


def test_merge_chunk_results_sorts_out_of_order_chunks() -> None:
    results = [
        ChunkResult(30.0, "en", 0.9, [TranscriptSegment(start=1.0, end=3.0, text="second")]),
        ChunkResult(0.0, "en", 0.9, [TranscriptSegment(start=1.0, end=3.0, text="first")]),
    ]

    assert [segment.text for segment in merge_chunk_results(results)] == ["first", "second"]


def test_detect_language_prefers_the_most_confident_chunk() -> None:
    results = [
        ChunkResult(0.0, "de", 0.41, []),
        ChunkResult(30.0, "en", 0.95, []),
    ]

    assert detect_language(results) == ("en", 0.95)


def test_transcribe_chunk_clamps_timestamps_to_the_chunk_and_drops_empty_text() -> None:
    model = FakeWhisperModel(results=[[FakeSegment(0.2, 4.5, "over the end"), FakeSegment(1.0, 2.0, "   ")]])
    service = TranscriptionService(settings=_test_settings(), model=model)
    chunk = AudioChunk(path=Path("chunk_0002.wav"), start_offset_seconds=10.0, duration_seconds=0.9)

    result = service._transcribe_chunk_with_retry(chunk)

    assert [(segment.start, segment.end, segment.text) for segment in result.segments] == [
        (0.2, 0.9, "over the end")
    ]


def test_transcribe_chunk_retries_then_succeeds() -> None:
    model = FakeWhisperModel(results=[[FakeSegment(0.0, 1.0, "Hi")]], failures=2)
    service = TranscriptionService(settings=_test_settings(), model=model)

    result = service._transcribe_chunk_with_retry(AudioChunk(path=Path("chunk_0000.wav"), start_offset_seconds=0.0, duration_seconds=30.0))

    assert len(model.calls) == 3
    assert result.segments[0].text == "Hi"


def test_transcribe_chunk_raises_after_exhausting_retries() -> None:
    model = FakeWhisperModel(failures=99)
    service = TranscriptionService(settings=_test_settings(), model=model)

    with pytest.raises(TranscriptionError, match="after 3 attempt"):
        service._transcribe_chunk_with_retry(AudioChunk(path=Path("chunk_0000.wav"), start_offset_seconds=0.0, duration_seconds=30.0))

    assert len(model.calls) == 3


@requires_ffmpeg
def test_pipeline_merges_chunk_timestamps_end_to_end(make_audio_file) -> None:
    source: Path = make_audio_file(duration_seconds=7.0, suffix=".mp3")
    model = FakeWhisperModel(
        results=[
            [FakeSegment(0.5, 2.0, "Hello")],
            [FakeSegment(1.0, 3.0, "How are you?")],
            [FakeSegment(0.2, 0.8, "Goodbye")],
        ]
    )
    service = TranscriptionService(settings=_test_settings(chunk_duration_seconds=3), model=model)

    response = service.transcribe_sync(source)

    assert response.chunk_count == 3
    assert response.language == "en"
    assert response.duration_seconds == pytest.approx(7.0, abs=0.2)
    assert [(segment.start, segment.end, segment.text) for segment in response.segments] == [
        (0.5, 2.0, "Hello"),
        (4.0, 6.0, "How are you?"),
        (6.2, 6.8, "Goodbye"),
    ]


def _test_settings(chunk_duration_seconds: int = 30) -> Settings:
    return Settings(
        preload_model=False,
        retry_delay_seconds=0.0,
        max_transcription_retries=3,
        chunk_duration_seconds=chunk_duration_seconds,
    )
