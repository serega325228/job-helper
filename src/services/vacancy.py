import asyncio
from uuid import UUID

from pydantic import ValidationError

from src.config.retry import RetryableLlmError, llm_retry
from src.infrastructure.db.sqlalchemy_unit_of_work import SqlAlchemyUnitOfWork
from src.infrastructure.llm.vacancy_analyzer import NormalizedVacancy, VacancyAnalyzer
from src.infrastructure.models.vacancy import Vacancy
from src.infrastructure.vacancy_sources.hh.schemas import NormalizedVacancyBatch, RawVacancy, VacancyReference
from src.ports.vacancy_source import VacancySource


class VacancyService:
    def __init__(
        self,
        unit_of_work: SqlAlchemyUnitOfWork,
        analyzer: VacancyAnalyzer
    ):
        self._uow = unit_of_work
        self._analyzer = analyzer

    async def create(
        self,
        name: str,
        raw_story: str,
    ):
        vacancy = Vacancy(
            name=name,
            raw_story=raw_story
        )

        async with self._uow as uow:
            await uow.vacancies.create(vacancy)

        return vacancy

    async def normalize_vacancies(
        self,
        vacancies: list[RawVacancy],
    ) -> list[NormalizedVacancy]:
        if not vacancies:
            return []

        try:
            normalized = await self._normalize_with_retry(vacancies)
        except (RetryableLlmError, ValidationError) as error:
            raise VacancyNormalizationError("Vacancy batch normalization failed") from error

        return self._validate_batch_consistency(
            raw=vacancies,
            normalized=normalized
        )

    @llm_retry()
    async def _normalize_with_retry(
        self,
        vacancies: list[RawVacancy],
    ) -> list[NormalizedVacancy]:
        try:
            return await self._analyzer.normalize(vacancies)
        except ValidationError:
            raise
        except TimeoutError as error:
            raise RetryableLlmError("LLM timeout") from error

    @staticmethod
    def _validate_batch_consistency(
        raw: list[RawVacancy],
        normalized: list[NormalizedVacancy],
    ) -> list[NormalizedVacancy]:
        raw_ids = {
            (vacancy.source, vacancy.external_id)
            for vacancy in raw
        }

        normalized_ids = {
            (vacancy.source, vacancy.external_id)
            for vacancy in normalized
        }

        missing = raw_ids - normalized_ids
        unexpected = normalized_ids - raw_ids

        if missing or unexpected:
            raise VacancyNormalizationError(
                f"Invalid batch mapping: "
                f"missing={missing}, unexpected={unexpected}"
            )

        if len(normalized_ids) != len(normalized):
            raise VacancyNormalizationError(
                "LLM returned duplicate vacancies"
            )

        return normalized

    async def fetch_vacancies(
        self,
        source: VacancySource,
        references: list[VacancyReference],
        concurrency: int = 8,
    ) -> list[RawVacancy]:
        semaphore = asyncio.Semaphore(concurrency)

        async def fetch_one(
            reference: VacancyReference,
        ) -> RawVacancy:
            async with semaphore:
                return await source.fetch_details(reference)

        results = await asyncio.gather(
            *(fetch_one(reference) for reference in references),
            return_exceptions=True,
        )

        vacancies: list[RawVacancy] = []

        for reference, result in zip(references, results, strict=True):
            if isinstance(result, Exception):
                # Записываем ошибку и статус для конкретной вакансии.
                continue

            vacancies.append(result)

        return vacancies
