from datetime import UTC, datetime, timedelta
from uuid import UUID

from dishka.integrations.taskiq import FromDishka, inject
from taskiq import TaskiqScheduler
from taskiq.schedule_sources import LabelScheduleSource

from src.config.settings import Settings, get_settings
from src.exceptions.profile import ProfileNotFoundError
from src.infrastructure.db.sqlalchemy_unit_of_work import SqlAlchemyUnitOfWork
from src.infrastructure.models.vacancy_match import VacancyMatch
from src.infrastructure.taskiq.broker import io_broker, laya_broker
from src.infrastructure.vacancy_sources.hh.source import HhVacancySource
from src.infrastructure.vacancy_sources.linkedin.source import LinkedInVacancySource
from src.schemas.vacancy import (
    BatchStatus,
    CollectionStatus,
    PreviewStatus,
    ProcessingStatus,
    RawVacancy,
    VacancyHardFilters,
    VacancyPreview,
    VacancyScrapingQuery,
    VacancyStatus,
)
from src.schemas.vacancy_match import MatchingCandidate
from src.services.scoring import ScoringService
from src.services.vacancy import VacancyService
from src.services.vacancy_match import VacancyMatchService
from src.services.vacancy_preview import VacancyPreviewService
from src.services.vacancy_scraping import PreviewServiceFactory, VacancyScrapingService


@io_broker.task(ack_type="when_executed")
@inject(patch_module=True)
async def collect_previews(
    collection_id: UUID,
    unit_of_work: FromDishka[SqlAlchemyUnitOfWork],
    preview_service_factory: FromDishka[PreviewServiceFactory],
    scraping: FromDishka[VacancyScrapingService],
    hh: FromDishka[HhVacancySource],
    linkedin: FromDishka[LinkedInVacancySource],
) -> None:
    async with unit_of_work as uow:
        collection = await uow.previews.get_collection(collection_id, lock=True)
        if collection is None or collection.status != CollectionStatus.PENDING:
            return

        async def save_batch(previews: list[VacancyPreview]) -> None:
            async with preview_service_factory() as service:
                saved = await service.save_scraped(collection.id, previews)
            filtered = saved[PreviewStatus.PENDING_FILTER]
            evaluated = saved[PreviewStatus.PENDING_LAYA]
            if filtered:
                await filter_previews.kiq(filtered)
            if evaluated:
                await evaluate_previews.kiq(evaluated)

        await scraping.scrape_previews(
            {"hh": hh, "linkedin": linkedin}[collection.source],
            collection.vacancy_page_url,
            VacancyScrapingQuery.model_validate(collection.query),
            VacancyHardFilters.model_validate(collection.hard_filters),
            save_batch,
        )
        collection.status = CollectionStatus.COMPLETED


@io_broker.task(ack_type="when_executed")
@inject(patch_module=True)
async def filter_previews(
    preview_ids: list[UUID], service: FromDishka[VacancyPreviewService]
) -> None:
    await service.evaluate(preview_ids, use_laya=False)


@laya_broker.task(ack_type="when_executed")
@inject(patch_module=True)
async def evaluate_previews(
    preview_ids: list[UUID], service: FromDishka[VacancyPreviewService]
) -> None:
    await service.evaluate(preview_ids, use_laya=True)


@io_broker.task(ack_type="when_executed")
@inject(patch_module=True)
async def scrape_details(
    vacancy_ids: list[UUID],
    unit_of_work: FromDishka[SqlAlchemyUnitOfWork],
    scraping: FromDishka[VacancyScrapingService],
    hh: FromDishka[HhVacancySource],
    linkedin: FromDishka[LinkedInVacancySource],
    settings: FromDishka[Settings],
) -> None:
    parsed_ids = []
    batch_ids = set()
    async with unit_of_work as uow:
        vacancies = await uow.vacancies.get_stage(
            vacancy_ids, ProcessingStatus.PENDING_SCRAPE
        )
        if not vacancies:
            return
        previews = {
            preview.id: VacancyPreview.model_validate(preview)
            for preview in await uow.previews.get_by_ids(
                [vacancy.preview_id for vacancy in vacancies]
            )
        }
        sources = {"hh": hh, "linkedin": linkedin}
        for source_name in sorted({vacancy.source for vacancy in vacancies}):
            group = [vacancy for vacancy in vacancies if vacancy.source == source_name]
            raw_by_key = await scraping.scrape_details(
                sources[source_name],
                [previews[vacancy.preview_id] for vacancy in group],
            )
            for vacancy in group:
                batch_ids.add(vacancy.batch_id)
                raw = raw_by_key[(vacancy.source, vacancy.external_id)]
                if raw is None:
                    vacancy.status = VacancyStatus.REMOVED
                    vacancy.processing_status = ProcessingStatus.REMOVED
                    continue
                vacancy.raw_document = raw.model_dump(mode="json")
                vacancy.raw_payload = raw.raw_payload
                vacancy.description = raw.raw_text or ""
                vacancy.processing_status = ProcessingStatus.PENDING_PARSE
                parsed_ids.append(vacancy.id)
    if parsed_ids:
        for offset in range(
            0, len(parsed_ids), settings.scraping.normalization_batch_size
        ):
            await parse_vacancies.kiq(
                parsed_ids[offset : offset + settings.scraping.normalization_batch_size]
            )
    for batch_id in batch_ids:
        await score_embeddings.kiq(batch_id)


@io_broker.task(ack_type="when_executed")
@inject(patch_module=True)
async def parse_vacancies(
    vacancy_ids: list[UUID],
    unit_of_work: FromDishka[SqlAlchemyUnitOfWork],
    service: FromDishka[VacancyService],
) -> None:
    batch_ids = set()
    async with unit_of_work as uow:
        vacancies = await uow.vacancies.get_stage(
            vacancy_ids, ProcessingStatus.PENDING_PARSE
        )
        if not vacancies:
            return
        raw = [RawVacancy.model_validate(vacancy.raw_document) for vacancy in vacancies]
        normalized = await service.normalize_vacancies(raw)
        normalized_by_key = {
            (item.source, item.external_id): item for item in normalized
        }
        for vacancy, document in zip(vacancies, raw, strict=True):
            values = service._to_persistence_values(
                raw=document,
                normalized=normalized_by_key[(vacancy.source, vacancy.external_id)],
                seen_at=datetime.now(UTC),
            )
            for field, value in values.items():
                setattr(vacancy, field, value)
            vacancy.processing_status = ProcessingStatus.PENDING_EMBEDDING
            batch_ids.add(vacancy.batch_id)
    for batch_id in batch_ids:
        await score_embeddings.kiq(batch_id)


@io_broker.task(ack_type="when_executed")
@inject(patch_module=True)
async def score_embeddings(
    batch_id: UUID,
    unit_of_work: FromDishka[SqlAlchemyUnitOfWork],
    service: FromDishka[VacancyService],
    scoring: FromDishka[ScoringService],
) -> None:
    selected_ids = []
    async with unit_of_work as uow:
        batch = await uow.vacancies.get_batch(batch_id)
        if batch is None or batch.status != BatchStatus.DETAILS:
            return
        vacancies = await uow.vacancies.get_batch_vacancies(batch_id)
        if any(
            vacancy.processing_status
            in {ProcessingStatus.PENDING_SCRAPE, ProcessingStatus.PENDING_PARSE}
            for vacancy in vacancies
        ):
            return
        pending = [
            vacancy
            for vacancy in vacancies
            if vacancy.processing_status == ProcessingStatus.PENDING_EMBEDDING
        ]
        if not pending:
            batch.status = BatchStatus.COMPLETED
            return
        profile = await uow.profiles.get_by_id(batch.profile_id)
        if profile is None:
            raise ProfileNotFoundError(batch.profile_id)
        preferences = [
            preference
            for preference in await uow.profiles.get_preferences_by_ids(
                batch.profile_id, [UUID(value) for value in batch.preference_ids]
            )
            if preference.enabled
        ]
        if not preferences:
            raise ValueError("No enabled preferences remain for this batch")
        await scoring.update_preference_embeddings(
            [
                preference
                for preference in preferences
                if preference.title_embedding is None
                or preference.content_embedding is None
            ]
        )
        raw_by_key = {
            (vacancy.source, vacancy.external_id): RawVacancy.model_validate(
                vacancy.raw_document
            )
            for vacancy in pending
        }
        embeddings = await service._build_embeddings(raw_by_key, pending)
        for vacancy in pending:
            vacancy.content_embedding, vacancy.title_embedding = embeddings[
                (vacancy.source, vacancy.external_id)
            ]
        await uow.flush()
        filters = VacancyHardFilters.model_validate(batch.hard_filters)
        results = await uow.vacancy_matches.search_by_preferences(
            batch.profile_id,
            filters,
            limit=batch.search_limit,
            title_weight=batch.title_weight,
            candidate_limit=len(pending),
            per_preference_limit=len(pending),
            vacancy_ids=[vacancy.id for vacancy in pending],
            preference_ids=[preference.id for preference in preferences],
        )
        by_id = {vacancy.id: vacancy for vacancy in pending}
        preferences_by_id = {preference.id: preference for preference in preferences}
        results_by_id = {result.vacancy_id: result for result in results}
        for vacancy in pending:
            result = results_by_id.get(vacancy.id)
            preference = (
                preferences_by_id[result.preference_id]
                if result
                else max(
                    preferences,
                    key=lambda preference: (
                        scoring.compare_preference(preference, vacancy).score
                    ),
                )
            )
            candidate = MatchingCandidate(
                preference_id=preference.id,
                title_similarity=result.title_similarity if result else 0,
                content_similarity=result.content_similarity if result else 0,
                embedding_similarity=result.combined_similarity if result else 0,
                profile_comparison=scoring.compare_profile(profile, vacancy),
                preference_comparison=scoring.compare_preference(preference, vacancy),
            )
            vacancy.scores = candidate.model_dump(mode="json")
        for result in results:
            vacancy = by_id[result.vacancy_id]
            if (
                not vacancy.scores["preference_comparison"]["hard_constraints_passed"]
                or len(selected_ids) >= batch.rerank_limit
            ):
                continue
            selected_ids.append(vacancy.id)
        for vacancy in pending:
            vacancy.processing_status = (
                ProcessingStatus.PENDING_RERANK
                if vacancy.id in selected_ids
                else ProcessingStatus.PENDING_LAYA
            )
        batch.status = (
            BatchStatus.PENDING_RERANK if selected_ids else BatchStatus.PENDING_LAYA
        )
    if selected_ids:
        await rerank_vacancies.kiq(batch_id)
    else:
        await evaluate_vacancies.kiq(batch_id)


@io_broker.task(ack_type="when_executed")
@inject(patch_module=True)
async def rerank_vacancies(
    batch_id: UUID,
    unit_of_work: FromDishka[SqlAlchemyUnitOfWork],
    scoring: FromDishka[ScoringService],
) -> None:
    async with unit_of_work as uow:
        batch = await uow.vacancies.get_batch(batch_id)
        if batch is None or batch.status != BatchStatus.PENDING_RERANK:
            return
        vacancies = [
            vacancy
            for vacancy in await uow.vacancies.get_batch_vacancies(batch_id)
            if vacancy.processing_status == ProcessingStatus.PENDING_RERANK
        ]
        if not vacancies:
            return
        profile = await uow.profiles.get_by_id(batch.profile_id)
        if profile is None:
            raise ProfileNotFoundError(batch.profile_id)
        preferences = {
            preference.id: preference
            for preference in await uow.profiles.get_preferences_by_ids(
                batch.profile_id, [UUID(value) for value in batch.preference_ids]
            )
        }
        candidates = {
            vacancy.id: MatchingCandidate.model_validate(vacancy.scores)
            for vacancy in vacancies
        }
        scores = await scoring.rerank_vacancies(
            profile,
            vacancies,
            {
                vacancy.id: preferences[candidates[vacancy.id].preference_id]
                for vacancy in vacancies
            },
        )
        for vacancy in vacancies:
            candidate = candidates[vacancy.id]
            candidate.profile_rerank_score = scores[vacancy.id].profile_score
            candidate.preference_rerank_score = scores[vacancy.id].preference_score
            vacancy.scores = candidate.model_dump(mode="json")
            vacancy.processing_status = ProcessingStatus.PENDING_LAYA
        batch.status = BatchStatus.PENDING_LAYA
    await evaluate_vacancies.kiq(batch_id)


@laya_broker.task(ack_type="when_executed")
@inject(patch_module=True)
async def evaluate_vacancies(
    batch_id: UUID,
    unit_of_work: FromDishka[SqlAlchemyUnitOfWork],
    scoring: FromDishka[ScoringService],
    settings: FromDishka[Settings],
) -> None:
    async with unit_of_work as uow:
        batch = await uow.vacancies.get_batch(batch_id)
        if batch is None or batch.status != BatchStatus.PENDING_LAYA:
            return
        vacancies = [
            vacancy
            for vacancy in await uow.vacancies.get_batch_vacancies(batch_id)
            if vacancy.processing_status == ProcessingStatus.PENDING_LAYA
        ]
        if not vacancies:
            return
        profile = await uow.profiles.get_by_id(batch.profile_id)
        if profile is None:
            raise ProfileNotFoundError(batch.profile_id)
        preferences = {
            preference.id: preference
            for preference in await uow.profiles.get_preferences_by_ids(
                batch.profile_id, [UUID(value) for value in batch.preference_ids]
            )
        }
        candidates = {
            vacancy.id: MatchingCandidate.model_validate(vacancy.scores)
            for vacancy in vacancies
        }
        evaluations = await scoring.laya_match_vacancies(
            profile,
            {
                vacancy.id: preferences[candidates[vacancy.id].preference_id]
                for vacancy in vacancies
            },
            {vacancy.id: vacancy for vacancy in vacancies},
            batch_size=settings.scraping.laya_batch_size,
        )
        for vacancy in vacancies:
            candidate = candidates[vacancy.id]
            candidate.laya_comparison = evaluations[vacancy.id]
            vacancy.evaluation = evaluations[vacancy.id].model_dump(mode="json")
            vacancy.scores = candidate.model_dump(mode="json")
            vacancy.processing_status = (
                ProcessingStatus.PENDING_SAVE
                if candidate.profile_rerank_score is not None
                else ProcessingStatus.FILTERED
            )
        batch.status = (
            BatchStatus.PENDING_SAVE
            if any(
                vacancy.processing_status == ProcessingStatus.PENDING_SAVE
                for vacancy in vacancies
            )
            else BatchStatus.COMPLETED
        )
    if batch.status == BatchStatus.PENDING_SAVE:
        await save_vacancy_scores.kiq(batch_id)


@io_broker.task(ack_type="when_executed")
@inject(patch_module=True)
async def save_vacancy_scores(
    batch_id: UUID,
    unit_of_work: FromDishka[SqlAlchemyUnitOfWork],
    matching: FromDishka[VacancyMatchService],
) -> None:
    async with unit_of_work as uow:
        batch = await uow.vacancies.get_batch(batch_id)
        if batch is None or batch.status != BatchStatus.PENDING_SAVE:
            return
        vacancies = [
            vacancy
            for vacancy in await uow.vacancies.get_batch_vacancies(batch_id)
            if vacancy.processing_status == ProcessingStatus.PENDING_SAVE
        ]
        if not vacancies:
            return
        existing = {
            match.vacancy_id: match
            for match in await uow.vacancy_matches.get_by_profile_and_vacancy_ids(
                batch.profile_id, [vacancy.id for vacancy in vacancies]
            )
        }
        for vacancy in vacancies:
            candidate = MatchingCandidate.model_validate(vacancy.scores)
            result = matching.build_result(
                profile_id=batch.profile_id,
                vacancy_id=vacancy.id,
                preference_intent_id=candidate.preference_id,
                profile_comparison=candidate.profile_comparison,
                preference_comparison=candidate.preference_comparison,
                laya_comparison=candidate.laya_comparison,
                profile_rerank_score=candidate.profile_rerank_score,
                preference_rerank_score=candidate.preference_rerank_score,
                title_similarity=candidate.title_similarity,
                content_similarity=candidate.content_similarity,
                embedding_similarity=candidate.embedding_similarity,
            )
            values = result.model_dump(mode="python")
            values["category"] = result.category.value
            match = existing.get(vacancy.id)
            if match is None:
                uow.vacancy_matches.add(VacancyMatch(**values))
            else:
                for field, value in values.items():
                    setattr(match, field, value)
            vacancy.scores = result.model_dump(mode="json")
            vacancy.processing_status = ProcessingStatus.COMPLETED
        batch.status = BatchStatus.COMPLETED


@io_broker.task(
    ack_type="when_executed", schedule=[{"cron": get_settings().taskiq.recovery_cron}]
)
@inject(patch_module=True)
async def reconcile_processing(
    unit_of_work: FromDishka[SqlAlchemyUnitOfWork], settings: FromDishka[Settings]
) -> None:
    now = datetime.now(UTC)
    cutoff = now - timedelta(seconds=settings.taskiq.stale_timeout_seconds)
    limit = settings.taskiq.recovery_batch_size
    scheduled = []
    async with unit_of_work as uow:
        for batch in await uow.vacancies.stale_batches(cutoff, limit):
            if batch.status == BatchStatus.DETAILS:
                scheduled.append((score_embeddings, batch.id))
            elif batch.status == BatchStatus.PENDING_RERANK:
                scheduled.append((rerank_vacancies, batch.id))
            elif batch.status == BatchStatus.PENDING_LAYA:
                scheduled.append((evaluate_vacancies, batch.id))
            else:
                scheduled.append((save_vacancy_scores, batch.id))
            batch.updated_at = now
        for collection in await uow.previews.stale_collections(cutoff, limit):
            scheduled.append((collect_previews, collection.id))
            collection.updated_at = now
        groups: dict[PreviewStatus, list] = {}
        for preview in await uow.previews.stale_previews(cutoff, limit):
            groups.setdefault(preview.status, []).append(preview)
        for status, previews in groups.items():
            task = (
                evaluate_previews
                if status == PreviewStatus.PENDING_LAYA
                else filter_previews
            )
            for offset in range(0, len(previews), settings.scraping.laya_batch_size):
                scheduled.append(
                    (
                        task,
                        [
                            preview.id
                            for preview in previews[
                                offset : offset + settings.scraping.laya_batch_size
                            ]
                        ],
                    )
                )
            for preview in previews:
                preview.updated_at = now
        pending: dict[ProcessingStatus, list] = {}
        for vacancy in await uow.vacancies.stale_vacancies(cutoff, limit):
            pending.setdefault(vacancy.processing_status, []).append(vacancy)
        for status, vacancies in pending.items():
            task = (
                scrape_details
                if status == ProcessingStatus.PENDING_SCRAPE
                else parse_vacancies
            )
            batch_size = (
                settings.scraping.max_detail_pages
                if status == ProcessingStatus.PENDING_SCRAPE
                else settings.scraping.normalization_batch_size
            )
            for offset in range(0, len(vacancies), batch_size):
                scheduled.append(
                    (
                        task,
                        [
                            vacancy.id
                            for vacancy in vacancies[offset : offset + batch_size]
                        ],
                    )
                )
            for vacancy in vacancies:
                vacancy.updated_at = now
    for task, argument in scheduled:
        await task.kiq(argument)


scheduler = TaskiqScheduler(io_broker, [LabelScheduleSource(io_broker)])
