from uuid import UUID

from dishka.integrations.taskiq import FromDishka, inject
from taskiq import TaskiqScheduler
from taskiq.schedule_sources import LabelScheduleSource

from src.config.settings import Settings, get_settings
from src.infrastructure.taskiq.broker import broker
from src.schemas.vacancy import (
    BatchStatus,
    CollectionStatus,
    PreviewStatus,
    ProcessingStatus,
)
from src.services.vacancy import VacancyService
from src.services.vacancy_match import VacancyMatchService
from src.services.vacancy_preview import VacancyPreviewService
from src.services.vacancy_scraping import VacancyScrapingService

settings = get_settings()


@broker.task(ack_type="when_executed", queue_name=settings.taskiq.io_queue)
@inject(patch_module=True)
async def collect_previews(
    collection_id: UUID, service: FromDishka[VacancyScrapingService]
) -> None:
    await service.collect_previews(collection_id, filter_previews.kiq)


@broker.task(ack_type="when_executed", queue_name=settings.taskiq.io_queue)
@inject(patch_module=True)
async def filter_previews(
    preview_ids: list[UUID], service: FromDishka[VacancyPreviewService]
) -> None:
    await service.filter_previews(preview_ids)


@broker.task(ack_type="when_executed", queue_name=settings.taskiq.io_queue)
@inject(patch_module=True)
async def scrape_details(
    vacancy_ids: list[UUID],
    service: FromDishka[VacancyScrapingService],
    settings: FromDishka[Settings],
) -> None:
    parsed_ids, batch_ids = await service.process_details(vacancy_ids)
    batch_size = settings.scraping.normalization_batch_size
    for offset in range(0, len(parsed_ids), batch_size):
        await parse_vacancies.kiq(parsed_ids[offset : offset + batch_size])
    for batch_id in batch_ids:
        await score_embeddings.kiq(batch_id)


@broker.task(ack_type="when_executed", queue_name=settings.taskiq.io_queue)
@inject(patch_module=True)
async def parse_vacancies(
    vacancy_ids: list[UUID], service: FromDishka[VacancyService]
) -> None:
    for batch_id in await service.parse_vacancies(vacancy_ids):
        await score_embeddings.kiq(batch_id)


@broker.task(ack_type="when_executed", queue_name=settings.taskiq.io_queue)
@inject(patch_module=True)
async def score_embeddings(
    batch_id: UUID,
    service: FromDishka[VacancyService],
    matching: FromDishka[VacancyMatchService],
) -> None:
    if await matching.score_embeddings(batch_id, service):
        await rerank_vacancies.kiq(batch_id)


@broker.task(ack_type="when_executed", queue_name=settings.taskiq.io_queue)
@inject(patch_module=True)
async def rerank_vacancies(
    batch_id: UUID, matching: FromDishka[VacancyMatchService]
) -> None:
    if await matching.rerank_batch(batch_id):
        await evaluate_vacancies.kiq(batch_id)


@broker.task(ack_type="when_executed", queue_name=settings.taskiq.laya_queue)
@inject(patch_module=True)
async def evaluate_vacancies(
    batch_id: UUID,
    matching: FromDishka[VacancyMatchService],
    settings: FromDishka[Settings],
) -> None:
    if await matching.evaluate_batch(
        batch_id, batch_size=settings.scraping.laya_batch_size
    ):
        await save_vacancy_scores.kiq(batch_id)


@broker.task(ack_type="when_executed", queue_name=settings.taskiq.io_queue)
@inject(patch_module=True)
async def save_vacancy_scores(
    batch_id: UUID, matching: FromDishka[VacancyMatchService]
) -> None:
    await matching.save_batch_scores(batch_id)


@broker.task(
    ack_type="when_executed",
    queue_name=settings.taskiq.io_queue,
    schedule=[{"cron": settings.taskiq.recovery_cron}],
)
@inject(patch_module=True)
async def reconcile_processing(
    service: FromDishka[VacancyService], settings: FromDishka[Settings]
) -> None:
    claimed = await service.claim_stale_processing(settings)
    tasks = {
        CollectionStatus.PENDING: collect_previews,
        PreviewStatus.PENDING_FILTER: filter_previews,
        ProcessingStatus.PENDING_SCRAPE: scrape_details,
        ProcessingStatus.PENDING_PARSE: parse_vacancies,
        BatchStatus.DETAILS: score_embeddings,
        BatchStatus.PENDING_RERANK: rerank_vacancies,
        BatchStatus.PENDING_LAYA: evaluate_vacancies,
        BatchStatus.PENDING_SAVE: save_vacancy_scores,
    }
    batch_sizes = {
        PreviewStatus.PENDING_FILTER: settings.scraping.laya_batch_size,
        ProcessingStatus.PENDING_SCRAPE: settings.scraping.max_detail_pages,
        ProcessingStatus.PENDING_PARSE: settings.scraping.normalization_batch_size,
    }
    for status, entity_ids in claimed.items():
        task = tasks[status]
        if status in batch_sizes:
            batch_size = batch_sizes[status]
            for offset in range(0, len(entity_ids), batch_size):
                await task.kiq(entity_ids[offset : offset + batch_size])
        else:
            for entity_id in entity_ids:
                await task.kiq(entity_id)


scheduler = TaskiqScheduler(broker, [LabelScheduleSource(broker)])
