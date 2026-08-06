import asyncio
import logging
from datetime import UTC, datetime

from pydantic import ValidationError

from src.config.retry import RetryableLlmError, llm_retry
from src.exceptions.vacancy import VacancyNormalizationError
from src.infrastructure.db.sqlalchemy_unit_of_work import SqlAlchemyUnitOfWork
from src.infrastructure.models.vacancy import Vacancy
from src.ports.vacancy_normalizer import VacancyNormalizer
from src.ports.vacancy_source import VacancySource
from src.schemas.vacancy import (
    NormalizedVacancy,
    RawVacancy,
    VacancyReference,
    VacancySearchQuery,
)

logger = logging.getLogger(__name__)


class VacancyService:
    NORMALIZER_VERSION = "llm-v1"

    def __init__(
        self,
        unit_of_work: SqlAlchemyUnitOfWork,
        normalizer: VacancyNormalizer,
    ) -> None:
        self._uow = unit_of_work
        self._normalizer = normalizer

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
            normalized = await self._normalize_with_retry(vacancies)
        except (RetryableLlmError, ValidationError) as error:
            raise VacancyNormalizationError(
                "Vacancy batch normalization failed",
            ) from error

        return self._validate_batch_consistency(
            raw=vacancies,
            normalized=normalized,
        )

    @llm_retry()
    async def _normalize_with_retry(
        self,
        vacancies: list[RawVacancy],
    ) -> list[NormalizedVacancy]:
        try:
            return await self._normalizer.normalize(vacancies)
        except ValidationError:
            raise
        except TimeoutError as error:
            raise RetryableLlmError("LLM timeout") from error

    @staticmethod
    def _validate_batch_consistency(
        raw: list[RawVacancy],
        normalized: list[NormalizedVacancy],
    ) -> list[NormalizedVacancy]:
        raw_ids = {(vacancy.source, vacancy.external_id) for vacancy in raw}
        normalized_ids = {
            (vacancy.source, vacancy.external_id)
            for vacancy in normalized
        }

        missing = raw_ids - normalized_ids
        unexpected = normalized_ids - raw_ids
        if missing or unexpected:
            raise VacancyNormalizationError(
                "Invalid batch mapping: "
                f"missing={missing}, unexpected={unexpected}",
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
            (vacancy.source, vacancy.external_id): vacancy
            for vacancy in raw_vacancies
        }
        keys = {
            (vacancy.source, vacancy.external_id)
            for vacancy in normalized_vacancies
        }

        if missing_raw := keys - raw_by_key.keys():
            raise VacancyNormalizationError(
                f"Raw vacancies are missing for normalized keys: {missing_raw}",
            )

        async with self._uow as uow:
            existing = await uow.vacancies.get_by_external_keys(keys)
            existing_by_key = {
                (vacancy.source, vacancy.external_id): vacancy
                for vacancy in existing
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
    ) -> dict:
        soft_conditions = normalized.soft_conditions.model_dump(mode="json")

        return {
            "source": normalized.source,
            "external_id": normalized.external_id,
            "url": str(raw.url),
            "title": raw.title or normalized.title,
            "company_name": normalized.company_name,
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
            "completeness_score": normalized.soft_conditions.completeness_score,
            "normalizer_version": self.NORMALIZER_VERSION,
            "status": "active",
            "published_at": raw.published_at,
            "last_seen_at": seen_at,
        }
