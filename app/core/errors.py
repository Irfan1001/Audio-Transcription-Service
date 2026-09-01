class ServiceError(Exception):
    """Base error mapped to an HTTP response by the app's exception handler."""

    status_code: int = 500


class UnsupportedAudioFormatError(ServiceError):
    status_code = 415


class FileTooLargeError(ServiceError):
    status_code = 413


class EmptyAudioError(ServiceError):
    status_code = 400


class FFmpegNotFoundError(ServiceError):
    status_code = 503


class AudioProcessingError(ServiceError):
    status_code = 422


class ModelNotLoadedError(ServiceError):
    status_code = 503


class TranscriptionError(ServiceError):
    status_code = 500
