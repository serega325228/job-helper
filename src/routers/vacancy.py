from uuid import UUID

from dishka.integrations.fastapi import FromDishka, inject
from fastapi import APIRouter, HTTPException, Query

from src.exceptions.profile import ProfileNotFoundError
from src.exceptions.vacancy import VacancyScrapingError
from src.infrastructure.vacancy_sources.hh.source import HhVacancySource
from src.infrastructure.vacancy_sources.linkedin.source import LinkedInVacancySource
from src.schemas.vacancy import (
    PreviewCollectionRequest,
    PreviewResponse,
    PreviewSelectionRequest,
)
from src.services.vacancy_preview import VacancyPreviewService
from src.tasks.scraping import collect_previews, scrape_details

router = APIRouter(prefix="/vacancies", tags=["Vacancies"])


@router.post("/previews/collect", status_code=202)
@inject
async def collect(
    request: PreviewCollectionRequest,
    service: FromDishka[VacancyPreviewService],
    hh: FromDishka[HhVacancySource],
    linkedin: FromDishka[LinkedInVacancySource],
) -> dict[str, list[UUID]]:
    try:
        collection_ids = await service.prepare_collections(
            request, {"hh": hh, "linkedin": linkedin}
        )
    except ProfileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (ValueError, VacancyScrapingError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    for collection_id in collection_ids:
        await collect_previews.kiq(collection_id)
    return {"collection_ids": collection_ids}


@router.get("/previews/{profile_id}", response_model=list[PreviewResponse])
@inject
async def list_previews(
    profile_id: UUID,
    service: FromDishka[VacancyPreviewService],
    limit: int = Query(default=100, ge=1, le=1000),
) -> list:
    return await service.list_previews(profile_id, limit)


@router.post("/process", status_code=202)
@inject
async def process(
    request: PreviewSelectionRequest, service: FromDishka[VacancyPreviewService]
) -> dict[str, list[UUID]]:
    try:
        vacancy_ids = await service.select_for_details(request)
    except ProfileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    if vacancy_ids:
        await scrape_details.kiq(vacancy_ids)
    return {"vacancy_ids": vacancy_ids}
