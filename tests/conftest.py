"""Shared fixtures. The real Whisper model is never loaded or downloaded in tests."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

os.environ["PRELOAD_MODEL"] = "false"
os.environ["RETRY_DELAY_SECONDS"] = "0"

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import audio

requires_ffmpeg = pytest.mark.skipif(not audio.ffmpeg_available(), reason="FFmpeg is not installed")


@dataclass
class FakeSegment:
    start: float
    end: float
    text: str


@dataclass
class FakeInfo:
    language: str = "en"
    language_probability: float = 0.97


@dataclass
class FakeWhisperModel:
    """Stand-in for faster-whisper: returns canned segments, optionally failing first."""

    results: Sequence[Sequence[FakeSegment]] = field(default_factory=lambda: [[FakeSegment(0.5, 2.0, "Hello")]])
    failures: int = 0
    calls: list[str] = field(default_factory=list)

    def transcribe(self, audio_path: str) -> tuple[list[FakeSegment], FakeInfo]:
        self.calls.append(audio_path)
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError("temporary transcription failure")
        index = min(len(self.calls) - 1, len(self.results) - 1)
        return list(self.results[index]), FakeInfo()


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def make_audio_file(tmp_path: Path):
    """Generate a sine-tone audio file of a given duration and format via FFmpeg."""

    def _make(duration_seconds: float = 2.0, suffix: str = ".mp3", sample_rate: int = 44_100) -> Path:
        path = tmp_path / f"sample{suffix}"
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate={sample_rate}:duration={duration_seconds}",
                "-ac", "2",
                str(path),
            ],
            check=True,
            capture_output=True,
        )
        return path

    return _make


def probe(path: Path, entries: str) -> str:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", entries,
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()
