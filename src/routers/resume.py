from pathlib import Path
from typing import Annotated
from uuid import UUID

from dishka import FromDishka
from dishka.integrations.fastapi import inject
from fastapi import File, Form, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute, APIRouter
from starlette.exceptions import HTTPException
from structlog import get_logger

from src.exceptions.config import ConfigError, LLMError
from src.exceptions.profile import AnalysisError, ProfileNotFoundError
from src.exceptions.resume import ResumeError, ResumeTurnError
from src.infrastructure.models.resume import ResumeLanguage
from src.schemas.master_resume import (
    AnswerMasterResumeRequest,
    MasterResumeResponse,
    MasterResumeTurnResponse,
    StartMasterResumeRequest,
    master_resume_response,
)
from src.services.master_resume import MasterResumeService
from src.services.resume import ResumeService

logger = get_logger()
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def public_resume_error(error: Exception) -> HTTPException:
    if isinstance(error, ResumeError):
        return HTTPException(error.status_code, error.code)
    if isinstance(error, ProfileNotFoundError):
        return HTTPException(404, "profile_not_found")
    if isinstance(error, ConfigError):
        return HTTPException(503, "resume_provider_not_configured")
    if isinstance(error, (LLMError, AnalysisError)):
        cause = error.e if isinstance(error, AnalysisError) else error
        unavailable = isinstance(cause, ConfigError) or (
            isinstance(cause, LLMError)
            and cause.code in {"timeout", "rate_limited", "provider_unavailable"}
        )
        return HTTPException(503 if unavailable else 502, "resume_provider_failed")
    logger.warning("Master resume operation failed", code="resume_operation_unavailable")
    return HTTPException(503, "resume_operation_unavailable")


class MasterResumeRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def bounded_errors(request):
            try:
                return await handler(request)
            except (HTTPException, RequestValidationError):
                raise
            except ResumeTurnError as error:
                public = public_resume_error(error.error)
                public.headers = {"X-Resume-Thread-ID": str(error.thread_id)}
                raise public from None
            except Exception as error:
                raise public_resume_error(error) from None

        return bounded_errors


router = APIRouter(prefix="/resume", tags=["Resume"], route_class=MasterResumeRoute)


@router.post("/{profile_id}/master/threads", response_model=MasterResumeTurnResponse)
@inject
async def start_master_resume(
    profile_id: UUID,
    request: StartMasterResumeRequest,
    service: FromDishka[MasterResumeService],
) -> MasterResumeTurnResponse:
    return await service.start(profile_id, request)


@router.post(
    "/{profile_id}/master/threads/{thread_id}/answers",
    response_model=MasterResumeTurnResponse,
)
@inject
async def answer_master_resume(
    profile_id: UUID,
    thread_id: UUID,
    request: AnswerMasterResumeRequest,
    service: FromDishka[MasterResumeService],
) -> MasterResumeTurnResponse:
    return await service.answer(profile_id, thread_id, request.answer)


@router.post(
    "/{profile_id}/master/import", status_code=201, response_model=MasterResumeResponse
)
@inject
async def import_master_resume(
    profile_id: UUID,
    file: Annotated[UploadFile, File()],
    language: Annotated[ResumeLanguage, Form()],
    resume_service: FromDishka[ResumeService],
) -> MasterResumeResponse:
    filename = file.filename or ""
    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(415, "unsupported_resume_extension")
    # Never request an unbounded read, including for chunked requests.
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "resume_file_too_large")
    if not content:
        raise HTTPException(400, "resume_file_empty")
    resume = await resume_service.import_master_resume(
        profile_id=profile_id, filename=filename, content=content, language=language.value
    )
    return master_resume_response(resume)


@router.get("/{profile_id}/master", response_model=MasterResumeResponse)
@router.get("/{profile_id}/master/{resume_id}", response_model=MasterResumeResponse)
@inject
async def get_master_resume(
    profile_id: UUID,
    resume_service: FromDishka[ResumeService],
    resume_id: UUID | None = None,
) -> MasterResumeResponse:
    return master_resume_response(await resume_service.get_master(profile_id, resume_id))
