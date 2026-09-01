from pydantic import BaseModel, Field


class TranscriptSegment(BaseModel):
    """A single transcribed segment, with timestamps relative to the original audio."""

    start: float = Field(description="Segment start in seconds.")
    end: float = Field(description="Segment end in seconds.")
    text: str


class TranscriptionResponse(BaseModel):
    language: str
    language_probability: float | None = None
    duration_seconds: float
    chunk_count: int
    segments: list[TranscriptSegment]


class HealthResponse(BaseModel):
    status: str
    whisper_model: str
    model_loaded: bool
    ffmpeg_available: bool


class ServiceInfoResponse(BaseModel):
    service: str
    version: str
    supported_formats: list[str]
    docs_url: str


class ErrorResponse(BaseModel):
    detail: str
