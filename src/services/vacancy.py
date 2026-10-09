from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from pydantic import ValidationError

from src.config.settings import Settings
from src.exceptions.vacancy import VacancyNormalizationError
from src.infrastructure.db.sqlalchemy_unit_of_work import SqlAlchemyUnitOfWork
from src.ports.vacancy_normalizer import VacancyNormalizer
from src.schemas.vacancy import (
    BatchStatus,
    CollectionStatus,
    NormalizedVacancy,
    PreviewStatus,
    ProcessingStatus,
    RawVacancy,
    VacancyStatus,
)


class VacancyService:
    def __init__(
        self,
        unit_of_work: SqlAlchemyUnitOfWork,
        normalizer: VacancyNormalizer,
        settings: Settings,
    ) -> None:
        self._uow = unit_of_work
        self._normalizer = normalizer
        self._settings = settings

    async def parse_vacancies(self, vacancy_ids: list[UUID]) -> set[UUID]:
        token = uuid4()
        async with self._uow as uow:
            vacancies = await uow.vacancies.claim_details(
                vacancy_ids, ProcessingStatus.PENDING_PARSE, token, self._settings
            )
        claimed_ids = [vacancy.id for vacancy in vacancies]
        if vacancies:
            try:
                raw = [
                    RawVacancy.model_validate(vacancy.raw_document)
                    for vacancy in vacancies
                ]
                normalized = await self.normalize_vacancies(raw)
                by_key = {(item.source, item.external_id): item for item in normalized}
                values = {
                    vacancy.id: {
                        **self._to_persistence_values(
                            raw=document,
                            normalized=by_key[(vacancy.source, vacancy.external_id)],
                            seen_at=datetime.now(UTC),
                        ),
                        "processing_status": ProcessingStatus.COMPLETED,
                    }
                    for vacancy, document in zip(vacancies, raw, strict=True)
                }
                async with self._uow as uow:
                    await uow.vacancies.finish_details(values, token)
            except Exception as error:
                async with self._uow as uow:
                    await uow.vacancies.fail_details(
                        claimed_ids,
                        token,
                        error,
                        self._settings,
                    )
                raise
        async with self._uow as uow:
            return await uow.vacancies.ready_batch_ids(vacancy_ids)

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

    @staticmethod
    def _to_persistence_values(
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
            "content_embedding": None,
            "title_embedding": None,
            "status": VacancyStatus.ACTIVE,
            "published_at": raw.published_at,
            "last_seen_at": seen_at,
        }


class VacancyRecoveryService:
    def __init__(self, unit_of_work: SqlAlchemyUnitOfWork) -> None:
        self._uow = unit_of_work

    async def claim_stale_processing(self, settings: Settings) -> dict:
        now = datetime.now(UTC)
        cutoff = now - timedelta(seconds=settings.taskiq.stale_timeout_seconds)
        lease_cutoff = now - timedelta(seconds=settings.taskiq.lease_timeout_seconds)
        limit = settings.taskiq.recovery_batch_size
        claimed = {}
        async with self._uow as uow:
            groups = (
                (
                    await uow.previews.stale_collections(cutoff, lease_cutoff, limit),
                    "status",
                    CollectionStatus.FAILED,
                ),
                (
                    await uow.previews.stale_previews(cutoff, limit),
                    "status",
                    PreviewStatus.FAILED,
                ),
                (
                    await uow.vacancies.stale_vacancies(cutoff, lease_cutoff, limit),
                    "processing_status",
                    ProcessingStatus.FAILED,
                ),
                (
                    await uow.vacancies.stale_batches(cutoff, lease_cutoff, limit),
                    "status",
                    BatchStatus.FAILED,
                ),
            )
            for rows, status_field, failure_status in groups:
                for row in rows:
                    if hasattr(row, "processing_token"):
                        row.processing_token = None
                        if row.attempts >= settings.taskiq.stage_attempts:
                            setattr(row, status_field, failure_status)
                            row.error = "Stage attempts exhausted"
                            if failure_status is BatchStatus.FAILED:
                                await uow.vacancy_matches.fail_batch(row.id)
                            continue
                    claimed.setdefault(getattr(row, status_field), []).append(row.id)
                    row.updated_at = now
        return claimed
