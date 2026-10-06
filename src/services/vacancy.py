import asyncio
from datetime import UTC, datetime, timedelta
from uuid import UUID

from pydantic import ValidationError
from structlog import get_logger

from src.config.settings import Settings
from src.exceptions.vacancy import VacancyNormalizationError
from src.infrastructure.db.sqlalchemy_unit_of_work import SqlAlchemyUnitOfWork
from src.infrastructure.models.vacancy import Vacancy
from src.ports.embedder import Embedder
from src.ports.vacancy_normalizer import VacancyNormalizer
from src.ports.vacancy_source import VacancySource
from src.schemas.vacancy import (
    BatchStatus,
    CollectionStatus,
    NormalizedVacancy,
    PreviewStatus,
    ProcessingStatus,
    RawVacancy,
    VacancyHardFilters,
    VacancyReference,
    VacancySearchQuery,
)
from src.services.embedding_text import build_vacancy_search_text

logger = get_logger()


class VacancyService:
    def __init__(
        self,
        unit_of_work: SqlAlchemyUnitOfWork,
        normalizer: VacancyNormalizer,
        embedder: Embedder,
    ) -> None:
        self._uow = unit_of_work
        self._normalizer = normalizer
        self._embedder = embedder

    async def parse_vacancies(self, vacancy_ids: list[UUID]) -> set[UUID]:
        batch_ids = set()
        async with self._uow as uow:
            vacancies = await uow.vacancies.get_stage(
                vacancy_ids, ProcessingStatus.PENDING_PARSE
            )
            if not vacancies:
                return batch_ids
            raw = [
                RawVacancy.model_validate(vacancy.raw_document) for vacancy in vacancies
            ]
            normalized = await self.normalize_vacancies(raw)
            normalized_by_key = {
                (item.source, item.external_id): item for item in normalized
            }
            for vacancy, document in zip(vacancies, raw, strict=True):
                values = self._to_persistence_values(
                    raw=document,
                    normalized=normalized_by_key[(vacancy.source, vacancy.external_id)],
                    seen_at=datetime.now(UTC),
                )
                for field, value in values.items():
                    setattr(vacancy, field, value)
                vacancy.processing_status = ProcessingStatus.PENDING_EMBEDDING
                batch_ids.add(vacancy.batch_id)
        return batch_ids

    async def claim_stale_processing(
        self, settings: Settings
    ) -> dict[
        CollectionStatus | PreviewStatus | ProcessingStatus | BatchStatus, list[UUID]
    ]:
        now = datetime.now(UTC)
        cutoff = now - timedelta(seconds=settings.taskiq.stale_timeout_seconds)
        limit = settings.taskiq.recovery_batch_size
        claimed = {}
        async with self._uow as uow:
            groups = (
                (await uow.previews.stale_collections(cutoff, limit), "status"),
                (await uow.previews.stale_previews(cutoff, limit), "status"),
                (
                    await uow.vacancies.stale_vacancies(cutoff, limit),
                    "processing_status",
                ),
                (await uow.vacancies.stale_batches(cutoff, limit), "status"),
            )
            for rows, status_field in groups:
                for row in rows:
                    claimed.setdefault(getattr(row, status_field), []).append(row.id)
                    row.updated_at = now
        return claimed

    async def ingest_vacancies(
        self,
        source: VacancySource,
        query: VacancySearchQuery,
        *,
        limit: int | None = None,
        fetch_concurrency: int = 8,
        normalization_batch_size: int = 5,
    ) -> list[Vacancy]:
        """Fetch, normalize and idempotently persist vacancies from a source."""
        if limit is not None and limit < 1:
            raise ValueError("limit must be greater than zero")
        if fetch_concurrency < 1:
            raise ValueError("fetch_concurrency must be greater than zero")
        if normalization_batch_size < 1:
            raise ValueError("normalization_batch_size must be greater than zero")

        references = await self.search_vacancies(
            source,
            query,
            limit=limit,
        )
        raw_vacancies = await self.fetch_vacancies(
            source,
            references,
            concurrency=fetch_concurrency,
        )

        normalized_vacancies: list[NormalizedVacancy] = []
        for offset in range(0, len(raw_vacancies), normalization_batch_size):
            batch = raw_vacancies[offset : offset + normalization_batch_size]
            normalized_vacancies.extend(await self.normalize_vacancies(batch))

        return await self.save_vacancies(raw_vacancies, normalized_vacancies)

    async def search_vacancies(
        self,
        source: VacancySource,
        query: VacancySearchQuery,
        *,
        limit: int | None = None,
    ) -> list[VacancyReference]:
        if limit is not None and limit < 1:
            raise ValueError("limit must be greater than zero")

        references: list[VacancyReference] = []
        seen: set[tuple[str, str]] = set()

        async for reference in source.search(query):
            key = (reference.source, reference.external_id)
            if key in seen:
                continue

            seen.add(key)
            references.append(reference)

            if limit is not None and len(references) >= limit:
                break

        return references

    async def list_by_hard_filters(
        self,
        filters: VacancyHardFilters,
        *,
        limit: int = 100,
    ) -> list[Vacancy]:
        if limit < 1:
            raise ValueError("limit must be greater than zero")

        async with self._uow as uow:
            return await uow.vacancies.list_by_hard_filters(
                filters,
                limit=limit,
            )

    async def ids_by_hard_filters(
        self,
        filters: VacancyHardFilters,
        *,
        limit: int = 100,
    ) -> list[UUID]:
        if limit < 1:
            raise ValueError("limit must be greater than zero")

        async with self._uow as uow:
            return await uow.vacancies.ids_by_hard_filters(
                filters,
                limit=limit,
            )

    async def get_vacancies_by_ids(
        self,
        vacancy_ids: list[UUID],
    ) -> list[Vacancy]:
        async with self._uow as uow:
            return await uow.vacancies.get_vacancies_by_ids(vacancy_ids)

    async def fetch_vacancies(
        self,
        source: VacancySource,
        references: list[VacancyReference],
        *,
        concurrency: int = 8,
    ) -> list[RawVacancy]:
        if concurrency < 1:
            raise ValueError("concurrency must be greater than zero")

        semaphore = asyncio.Semaphore(concurrency)

        async def fetch_one(reference: VacancyReference) -> RawVacancy:
            async with semaphore:
                return await source.fetch_details(reference)

        results = await asyncio.gather(
            *(fetch_one(reference) for reference in references),
            return_exceptions=True,
        )

        vacancies: list[RawVacancy] = []
        for reference, result in zip(references, results, strict=True):
            if isinstance(result, Exception):
                logger.warning(
                    "Failed to fetch vacancy source=%s external_id=%s: %r",
                    reference.source,
                    reference.external_id,
                    result,
                )
                continue
            vacancies.append(result)

        return vacancies

    async def normalize_vacancies(
        self,
        vacancies: list[RawVacancy],
    ) -> list[NormalizedVacancy]:
        if not vacancies:
            return []

        try:
            normalized = await self._normalizer.normalize(vacancies)
        except ValidationError as error:
            raise VacancyNormalizationError(
                "Vacancy batch normalization failed",
            ) from error

        return self._validate_batch_consistency(
            raw=vacancies,
            normalized=normalized,
        )

    @staticmethod
    def _validate_batch_consistency(
        raw: list[RawVacancy],
        normalized: list[NormalizedVacancy],
    ) -> list[NormalizedVacancy]:
        raw_ids = {(vacancy.source, vacancy.external_id) for vacancy in raw}
        normalized_ids = {
            (vacancy.source, vacancy.external_id) for vacancy in normalized
        }

        missing = raw_ids - normalized_ids
        unexpected = normalized_ids - raw_ids
        if missing or unexpected:
            raise VacancyNormalizationError(
                f"Invalid batch mapping: missing={missing}, unexpected={unexpected}",
            )
        if len(normalized_ids) != len(normalized):
            raise VacancyNormalizationError(
                "Normalizer returned duplicate vacancies",
            )

        return normalized

    async def save_vacancies(
        self,
        raw_vacancies: list[RawVacancy],
        normalized_vacancies: list[NormalizedVacancy],
    ) -> list[Vacancy]:
        raw_by_key = {
            (vacancy.source, vacancy.external_id): vacancy for vacancy in raw_vacancies
        }
        keys = {
            (vacancy.source, vacancy.external_id) for vacancy in normalized_vacancies
        }

        if missing_raw := keys - raw_by_key.keys():
            raise VacancyNormalizationError(
                f"Raw vacancies are missing for normalized keys: {missing_raw}",
            )

        embeddings = await self._build_embeddings(
            raw_by_key,
            normalized_vacancies,
        )

        async with self._uow as uow:
            existing = await uow.vacancies.get_by_external_keys(keys)
            existing_by_key = {
                (vacancy.source, vacancy.external_id): vacancy for vacancy in existing
            }

            saved: list[Vacancy] = []
            new: list[Vacancy] = []
            seen_at = datetime.now(UTC)

            for normalized in normalized_vacancies:
                key = (normalized.source, normalized.external_id)
                values = self._to_persistence_values(
                    raw=raw_by_key[key],
                    normalized=normalized,
                    seen_at=seen_at,
                    content_embedding=embeddings[key][0],
                    title_embedding=embeddings[key][1],
                )
                vacancy = existing_by_key.get(key)

                if vacancy is None:
                    vacancy = Vacancy(**values)
                    new.append(vacancy)
                else:
                    for field, value in values.items():
                        setattr(vacancy, field, value)

                saved.append(vacancy)

            uow.vacancies.add_all(new)
            await uow.flush()

        return saved

    def _to_persistence_values(
        self,
        *,
        raw: RawVacancy,
        normalized: NormalizedVacancy,
        seen_at: datetime,
        content_embedding: list[float] | None = None,
        title_embedding: list[float] | None = None,
    ) -> dict:
        soft_conditions = normalized.soft_conditions.model_dump(mode="json")

        return {
            "source": normalized.source,
            "external_id": normalized.external_id,
            "url": str(raw.url),
            "title": raw.title or normalized.title,
            "company_name": normalized.company_name,
            "normalized_company": " ".join(
                (normalized.company_name or "").casefold().split()
            )
            or None,
            "normalized_title": " ".join(
                (raw.title or normalized.title).casefold().split()
            ),
            "description": normalized.description,
            "area_id": normalized.area_id,
            "country": normalized.country,
            "city": normalized.city,
            "work_format": normalized.work_format,
            "employment_type": normalized.employment_type,
            "work_schedule": normalized.work_schedule,
            "experience": normalized.experience,
            "seniority": normalized.seniority,
            "salary_from": normalized.salary_from,
            "salary_to": normalized.salary_to,
            "salary_currency": normalized.salary_currency,
            "salary_gross": normalized.salary_gross,
            "soft_conditions": soft_conditions,
            "raw_payload": raw.raw_payload,
            "content_embedding": content_embedding,
            "title_embedding": title_embedding,
            "status": "active",
            "published_at": raw.published_at,
            "last_seen_at": seen_at,
        }

    async def _build_embeddings(
        self,
        raw_by_key: dict[tuple[str, str], RawVacancy],
        normalized_vacancies: list[NormalizedVacancy],
    ) -> dict[tuple[str, str], tuple[list[float], list[float]]]:
        if not normalized_vacancies:
            return {}

        titles = [
            raw_by_key[(vacancy.source, vacancy.external_id)].title or vacancy.title
            for vacancy in normalized_vacancies
        ]
        content_documents = [
            f"{title}\n{build_vacancy_search_text(vacancy)}"
            for title, vacancy in zip(titles, normalized_vacancies, strict=True)
        ]
        vectors = await self._embedder.embed(content_documents + titles)
        split_at = len(normalized_vacancies)
        content_vectors = vectors[:split_at]
        title_vectors = vectors[split_at:]

        return {
            (vacancy.source, vacancy.external_id): (
                content_vector,
                title_vector,
            )
            for vacancy, content_vector, title_vector in zip(
                normalized_vacancies,
                content_vectors,
                title_vectors,
                strict=True,
            )
        }
