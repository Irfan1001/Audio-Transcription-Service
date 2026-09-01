from __future__ import annotations

from pathlib import Path

import pytest

from app.core.errors import AudioProcessingError, UnsupportedAudioFormatError
from app.services import audio
from tests.conftest import probe, requires_ffmpeg


@pytest.mark.parametrize("filename", ["a.wav", "a.MP3", "recording.m4a", "x.flac", "y.ogg"])
def test_validate_extension_accepts_supported_formats(filename: str) -> None:
    assert audio.validate_extension(filename).startswith(".")


@pytest.mark.parametrize("filename", ["notes.txt", "clip.mp4", "noextension", None])
def test_validate_extension_rejects_other_formats(filename: str | None) -> None:
    with pytest.raises(UnsupportedAudioFormatError):
        audio.validate_extension(filename)


@requires_ffmpeg
def test_normalize_audio_produces_16khz_mono_pcm(make_audio_file, tmp_path: Path) -> None:
    source: Path = make_audio_file(duration_seconds=1.0, suffix=".mp3", sample_rate=44_100)

    normalized = audio.normalize_audio(source, tmp_path / "normalized.wav")

    assert normalized.exists()
    assert probe(normalized, "stream=sample_rate") == str(audio.TARGET_SAMPLE_RATE)
    assert probe(normalized, "stream=channels") == str(audio.TARGET_CHANNELS)
    assert probe(normalized, "stream=codec_name") == audio.TARGET_CODEC


@requires_ffmpeg
def test_normalize_audio_rejects_a_file_without_audio(tmp_path: Path) -> None:
    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"definitely not a wav file")

    with pytest.raises(AudioProcessingError):
        audio.normalize_audio(broken, tmp_path / "out.wav")


@requires_ffmpeg
def test_short_audio_is_not_split(make_audio_file, tmp_path: Path) -> None:
    source: Path = make_audio_file(duration_seconds=2.0, suffix=".wav")
    normalized = audio.normalize_audio(source, tmp_path / "normalized.wav")

    chunks = audio.split_into_chunks(normalized, chunk_duration_seconds=30, output_dir=tmp_path)

    assert len(chunks) == 1
    assert chunks[0].path == normalized
    assert chunks[0].start_offset_seconds == 0.0
    assert chunks[0].duration_seconds == pytest.approx(2.0, abs=0.1)


@requires_ffmpeg
def test_long_audio_is_split_with_increasing_offsets(make_audio_file, tmp_path: Path) -> None:
    source: Path = make_audio_file(duration_seconds=7.0, suffix=".wav")
    normalized = audio.normalize_audio(source, tmp_path / "normalized.wav")

    chunks = audio.split_into_chunks(normalized, chunk_duration_seconds=3, output_dir=tmp_path)

    assert [chunk.start_offset_seconds for chunk in chunks] == [0.0, 3.0, 6.0]
    assert all(chunk.path.exists() for chunk in chunks)
    assert audio.get_duration_seconds(chunks[0].path) == pytest.approx(3.0, abs=0.1)
    assert audio.get_duration_seconds(chunks[-1].path) == pytest.approx(1.0, abs=0.1)


def test_split_rejects_non_positive_chunk_duration(tmp_path: Path) -> None:
    with pytest.raises(AudioProcessingError):
        audio.split_into_chunks(tmp_path / "missing.wav", chunk_duration_seconds=0, output_dir=tmp_path)
