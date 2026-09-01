from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import app
from app.models.schemas import TranscriptionResponse
from app.services.transcription import TranscriptionService, get_transcription_service
from tests.conftest import FakeSegment, FakeWhisperModel, requires_ffmpeg


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False


def test_root_lists_supported_formats(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert ".mp3" in response.json()["supported_formats"]


def test_unsupported_file_type_is_rejected(client: TestClient) -> None:
    response = client.post("/transcribe", files={"file": ("notes.txt", b"not audio", "text/plain")})

    assert response.status_code == 415
    assert "Unsupported file type" in response.json()["detail"]


def test_missing_file_is_rejected(client: TestClient) -> None:
    assert client.post("/transcribe").status_code == 422


def test_empty_file_is_rejected(client: TestClient) -> None:
    response = client.post("/transcribe", files={"file": ("sample.wav", b"", "audio/wav")})

    assert response.status_code == 400


@requires_ffmpeg
def test_transcribe_without_loaded_model_returns_503(client: TestClient, make_audio_file) -> None:
    audio_path: Path = make_audio_file(duration_seconds=1.0, suffix=".wav")

    response = client.post("/transcribe", files={"file": ("sample.wav", audio_path.read_bytes(), "audio/wav")})

    assert response.status_code == 503


@requires_ffmpeg
def test_transcribe_returns_expected_response_structure(make_audio_file) -> None:
    audio_path: Path = make_audio_file(duration_seconds=4.0, suffix=".mp3")
    fake_service = TranscriptionService(
        settings=Settings(preload_model=False, retry_delay_seconds=0.0),
        model=FakeWhisperModel(results=[[FakeSegment(0.52, 3.41, " Hello, how are you? ")]]),
    )
    app.dependency_overrides[get_transcription_service] = lambda: fake_service

    try:
        with TestClient(app) as client:
            response = client.post(
                "/transcribe", files={"file": ("sample.mp3", audio_path.read_bytes(), "audio/mpeg")}
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = TranscriptionResponse.model_validate(response.json())
    assert body.language == "en"
    assert body.chunk_count == 1
    assert [(segment.start, segment.end, segment.text) for segment in body.segments] == [
        (0.52, 3.41, "Hello, how are you?")
    ]
