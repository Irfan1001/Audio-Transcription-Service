import logging
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile

from app import __version__
from app.core.config import Settings, get_settings
from app.models.schemas import (
    ErrorResponse,
    HealthResponse,
    ServiceInfoResponse,
    TranscriptionResponse,
)
from app.services import audio
from app.services.transcription import TranscriptionService, get_transcription_service

logger = logging.getLogger(__name__)

router = APIRouter()

SettingsDep = Annotated[Settings, Depends(get_settings)]
ServiceDep = Annotated[TranscriptionService, Depends(get_transcription_service)]


@router.get("/", response_model=ServiceInfoResponse)
async def service_info() -> ServiceInfoResponse:
    return ServiceInfoResponse(
        service="Audio Transcription Service",
        version=__version__,
        supported_formats=list(audio.SUPPORTED_EXTENSIONS),
        docs_url="/docs",
    )


@router.get("/health", response_model=HealthResponse)
async def health(settings: SettingsDep, service: ServiceDep) -> HealthResponse:
    return HealthResponse(
        status="ok",
        whisper_model=settings.whisper_model,
        model_loaded=service.model_loaded,
        ffmpeg_available=audio.ffmpeg_available(),
    )


@router.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    responses={
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def transcribe(
    settings: SettingsDep,
    service: ServiceDep,
    file: Annotated[UploadFile, File(description="Audio file to transcribe.")],
) -> TranscriptionResponse:
    """Transcribe an uploaded audio file and return segments with timestamps."""
    logger.info("Upload requested: filename=%s content_type=%s", file.filename, file.content_type)

    with tempfile.TemporaryDirectory(prefix="upload-") as tmpdir:
        source = await audio.save_upload(file, Path(tmpdir), settings.max_file_size_bytes)
        return await service.transcribe(source)
