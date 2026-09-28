"""Liveness and independent configuration, LLM, and database status checks."""

import logging
from uuid import UUID

from dishka.integrations.fastapi import FromDishka, inject
from fastapi import APIRouter

from src.infrastructure.db.sqlalchemy_unit_of_work import SqlAlchemyUnitOfWork
from src.infrastructure.llm.llm import LLMConfigManager, LLMProvider
from src.schemas.api import HealthResponse, StatusResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(status="healthy")


@router.get("/status/{profile_id}", response_model=StatusResponse)
@inject
async def get_status(
    config_manager: FromDishka[LLMConfigManager],
    llm: FromDishka[LLMProvider],
    uow: FromDishka[SqlAlchemyUnitOfWork],
    profile_id: UUID,
) -> StatusResponse:
    llm_configured = llm.configured
    features_configured = config_manager.get_features() is not None
    llm_healthy = False
    llm_error_code = None
    if llm_configured:
        try:
            health = await llm.check_llm_health()
            llm_healthy = health.healthy
            llm_error_code = health.error_code
        except Exception:
            logger.exception("Status: LLM health probe failed")
            llm_error_code = "health_check_failed"
    else:
        llm_error_code = "configuration_required"

    database_healthy = False
    db_stats = {
        "total_resumes": 0,
        "total_scrapped_vacancies": 0,
        "total_matched_vacancies": 0,
        "has_master_resume": False,
    }
    try:
        db_stats = await uow.get_stats(profile_id)
        database_healthy = True
    except Exception:
        logger.exception("Status: database stats failed")
    has_master_resume = bool(db_stats.get("has_master_resume"))
    if not llm_configured or not features_configured:
        status = "setup_required"
    elif not database_healthy or not llm_healthy:
        status = "degraded"
    elif not has_master_resume:
        status = "setup_required"
    else:
        status = "ready"
    return StatusResponse(
        status=status,
        llm_configured=llm_configured,
        features_configured=features_configured,
        llm_healthy=llm_healthy,
        llm_error_code=llm_error_code,
        database_healthy=database_healthy,
        has_master_resume=has_master_resume,
        database_stats=db_stats,
    )
