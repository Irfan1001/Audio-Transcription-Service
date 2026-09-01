import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app import __version__
from app.api.routes import router
from app.core.config import get_settings
from app.core.errors import ServiceError
from app.core.logging_config import configure_logging
from app.services import audio
from app.services.transcription import get_transcription_service

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Configure logging and load the Whisper model once at startup."""
    settings = get_settings()
    configure_logging(settings.log_level)

    if not audio.ffmpeg_available():
        logger.warning("FFmpeg/ffprobe not found on PATH: /transcribe will fail until FFmpeg is installed.")

    if settings.preload_model:
        get_transcription_service().load_model()
    else:
        logger.warning("PRELOAD_MODEL is disabled: /transcribe will report the model as unavailable.")

    yield


app = FastAPI(
    title="Audio Transcription Service",
    description="Upload an audio file and get a Whisper transcription with per-segment timestamps.",
    version=__version__,
    lifespan=lifespan,
)
app.include_router(router)


@app.exception_handler(ServiceError)
async def service_error_handler(request: Request, exc: ServiceError) -> JSONResponse:
    logger.error("%s on %s: %s", type(exc).__name__, request.url.path, exc)
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})
