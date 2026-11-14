from uuid import UUID

from dishka import FromDishka
from dishka.integrations.fastapi import inject
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute, APIRouter
from starlette.exceptions import HTTPException
from structlog import get_logger

from src.exceptions.config import ConfigError, LLMError
from src.exceptions.profile import (
    AnalysisError,
    ProfileNotFoundError,
    ProfileTurnError,
    ProfileWorkflowError,
)
from src.exceptions.resume import ResumeError, ResumeTurnError
from src.infrastructure.models.resume import ResumeLanguage
from src.schemas.master_resume import MasterResumeTurnResponse, StartMasterResumeRequest
from src.schemas.wizard import (
    AnswerWizardRequest,
    PreferenceTitleCatalogResponse,
    StartProfileWizardRequest,
    StartWizardRequest,
    WizardTurnResponse,
)
from src.services.master_resume import MasterResumeService
from src.services.preference_catalog import CATALOG_VERSION, ROLE_TITLES
from src.services.wizard import WizardService

logger = get_logger()


def public_wizard_error(error: Exception) -> HTTPException:
    if isinstance(error, (ProfileWorkflowError, ResumeError)):
        return HTTPException(error.status_code, error.code)
    if isinstance(error, ProfileNotFoundError):
        return HTTPException(404, "profile_not_found")
    if isinstance(error, ConfigError):
        return HTTPException(503, "wizard_provider_not_configured")
    if isinstance(error, (LLMError, AnalysisError)):
        cause = error.e if isinstance(error, AnalysisError) else error
        unavailable = isinstance(cause, ConfigError) or (
            isinstance(cause, LLMError)
            and cause.code in {"timeout", "rate_limited", "provider_unavailable"}
        )
        return HTTPException(503 if unavailable else 502, "wizard_provider_failed")
    logger.warning("Story wizard operation failed", code="wizard_operation_unavailable")
    return HTTPException(503, "wizard_operation_unavailable")


class WizardRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def bounded_errors(request):
            try:
                return await handler(request)
            except (HTTPException, RequestValidationError):
                raise
            except ProfileTurnError as error:
                public = public_wizard_error(error.error)
                public.headers = {
                    "X-Wizard-Profile-ID": str(error.profile_id),
                    "X-Wizard-Thread-ID": str(error.thread_id),
                }
                raise public from None
            except ResumeTurnError as error:
                public = public_wizard_error(error.error)
                public.headers = {"X-Resume-Thread-ID": str(error.thread_id)}
                raise public from None
            except Exception as error:
                raise public_wizard_error(error) from None

        return bounded_errors


router = APIRouter(prefix="/wizard", tags=["Wizard"], route_class=WizardRoute)


@router.get("/preference-titles", response_model=PreferenceTitleCatalogResponse)
async def preference_titles() -> PreferenceTitleCatalogResponse:
    return PreferenceTitleCatalogResponse(version=CATALOG_VERSION, roles=ROLE_TITLES)


@router.post("/threads", response_model=WizardTurnResponse, status_code=201)
@inject
async def start_wizard(
    request: StartWizardRequest,
    service: FromDishka[WizardService],
) -> WizardTurnResponse:
    return await service.start(request)


@router.post("/{profile_id}/threads", response_model=WizardTurnResponse, status_code=201)
@inject
async def start_profile_wizard(
    profile_id: UUID,
    request: StartProfileWizardRequest,
    service: FromDishka[WizardService],
) -> WizardTurnResponse:
    return await service.start_profile(profile_id, request)


@router.get("/{profile_id}/threads/{thread_id}", response_model=WizardTurnResponse)
@inject
async def get_wizard(
    profile_id: UUID, thread_id: UUID, service: FromDishka[WizardService],
) -> WizardTurnResponse:
    return await service.get(profile_id, thread_id)


@router.post("/{profile_id}/threads/{thread_id}/answers", response_model=WizardTurnResponse)
@inject
async def answer_wizard(
    profile_id: UUID, thread_id: UUID, request: AnswerWizardRequest,
    service: FromDishka[WizardService],
) -> WizardTurnResponse:
    return await service.answer(profile_id, thread_id, request)


@router.post("/{profile_id}/threads/{thread_id}/master", response_model=MasterResumeTurnResponse)
@inject
async def create_master_from_wizard(
    profile_id: UUID, thread_id: UUID,
    service: FromDishka[WizardService], master_service: FromDishka[MasterResumeService],
) -> MasterResumeTurnResponse:
    language = await service.ensure_master_available(profile_id, thread_id)
    # MasterResumeService rechecks uniqueness under the same profile turn guard.
    return await master_service.start(profile_id, StartMasterResumeRequest(
        language=ResumeLanguage(language), creation_method="profile",
    ))
