import re

from src.exceptions.vacancy import VacancyPreviewEvaluationError
from src.infrastructure.laya.laya_provider import LayaProvider
from src.infrastructure.laya.questions import MATCH_QUESTIONS
from src.infrastructure.models.preference_intent import PreferenceIntent
from src.infrastructure.models.profile import Profile
from src.schemas.scoring import LayaComparison
from src.schemas.vacancy import (
    VacancyHardFilters,
    VacancyPreview,
    VacancyScrapingQuery,
    WorkFormat,
)
from src.services.embedding_text import build_preference_title_text
from src.services.improver import ImproverService
from src.services.scoring import EXPERIENCE_MINIMUMS, SENIORITY_RANKS


def _normalize(value: str | None) -> str:
    return " ".join((value or "").casefold().split())


def _contains(value: str, text: str) -> bool:
    return ImproverService.keyword_in_text(_normalize(value), _normalize(text))


def _seniority_rank(value: str | None) -> int | None:
    ranks = {
        rank for name, rank in SENIORITY_RANKS.items() if _contains(name, value or "")
    }
    return ranks.pop() if len(ranks) == 1 else None


def _required_years(value: str | None) -> float | None:
    normalized = _normalize(value)
    native_minimum = EXPERIENCE_MINIMUMS.get(normalized)
    if native_minimum is not None:
        return native_minimum
    match = re.fullmatch(
        r"(?:at least\s+|minimum\s+|from\s+|от\s+|не менее\s+)?"
        r"(?P<minimum>\d+(?:[.,]\d+)?)\s*"
        r"(?:\+|[-–]\s*\d+(?:[.,]\d+)?)?\s*"
        r"(?:years?(?:\s+(?:of\s+)?experience)?|лет|года?)(?:\s+опыта)?",
        normalized,
    )
    return float(match["minimum"].replace(",", ".")) if match else None


class VacancyPreviewEvaluator:
    def __init__(self, laya: LayaProvider) -> None:
        self._laya = laya

    def accept(
        self,
        preview: VacancyPreview,
        profile: Profile,
        preferences: list[PreferenceIntent],
        filters: VacancyHardFilters,
        query: VacancyScrapingQuery,
    ) -> bool:
        if filters.sources and preview.source not in filters.sources:
            return False
        if filters.statuses and "active" not in filters.statuses:
            return False

        for expected, actual in (
            (filters.company_names, preview.company_name),
            (filters.work_formats, preview.work_format),
            (filters.employment_types, preview.employment_type),
            (filters.experience, preview.experience),
        ):
            if (
                expected
                and actual
                and _normalize(actual) not in map(_normalize, expected)
            ):
                return False
        if preview.company_name and _normalize(preview.company_name) in map(
            _normalize,
            filters.excluded_company_names,
        ):
            return False

        for locations in (filters.cities, filters.countries):
            if (
                locations
                and preview.location
                and not any(
                    _contains(location, preview.location) for location in locations
                )
            ):
                return False

        rank = _seniority_rank(preview.seniority or preview.title)
        allowed_ranks = [_seniority_rank(value) for value in filters.seniorities]
        if (
            rank is not None
            and allowed_ranks
            and None not in allowed_ranks
            and rank not in allowed_ranks
        ):
            return False
        required_years = _required_years(preview.experience)
        if (
            required_years is not None
            and profile.experience_years is not None
            and required_years > profile.experience_years
        ):
            return False
        if self._salary_fails(
            preview,
            filters.salary_min,
            filters.salary_currency,
            filters.salary_gross,
        ):
            return False

        published_at = preview.published_at
        for cutoff in (filters.published_after, query.published_after):
            if (
                published_at is not None
                and cutoff is not None
                and (published_at.tzinfo is None) == (cutoff.tzinfo is None)
                and published_at < cutoff
            ):
                return False

        text = "\n".join(
            value for value in (preview.title, preview.short_description) if value
        )
        if any(not _contains(keyword, text) for keyword in query.required_keywords):
            return False
        if any(_contains(keyword, text) for keyword in query.excluded_keywords):
            return False
        return not preferences or any(
            preference.enabled and self._matches_preference(preview, preference)
            for preference in preferences
        )

    @staticmethod
    def _salary_fails(
        preview: VacancyPreview,
        minimum: int | None,
        currency: str | None,
        gross: bool | None,
    ) -> bool:
        return (
            minimum is not None
            and preview.salary_to is not None
            and currency is not None
            and preview.salary_currency is not None
            and _normalize(currency) == _normalize(preview.salary_currency)
            and gross is not None
            and preview.salary_gross is gross
            and preview.salary_to < minimum
        )

    def _matches_preference(
        self,
        preview: VacancyPreview,
        preference: PreferenceIntent,
    ) -> bool:
        if any(_contains(title, preview.title) for title in preference.excluded_titles):
            return False
        if preview.company_name and _normalize(preview.company_name) in map(
            _normalize,
            preference.excluded_companies,
        ):
            return False
        for expected, actual in (
            (preference.work_formats, preview.work_format),
            (preference.employment_types, preview.employment_type),
        ):
            if (
                expected
                and actual
                and _normalize(actual) not in map(_normalize, expected)
            ):
                return False
        if (
            preference.locations
            and preview.location
            and preview.work_format in {WorkFormat.ON_SITE, WorkFormat.HYBRID}
            and not any(
                _contains(location, preview.location)
                for location in preference.locations
            )
        ):
            return False
        rank = _seniority_rank(preview.seniority or preview.title)
        minimum = _seniority_rank(preference.min_seniority)
        maximum = _seniority_rank(preference.max_seniority)
        return rank is None or (
            (minimum is None or rank >= minimum)
            and (maximum is None or rank <= maximum)
        )

    async def evaluate(
        self,
        previews: list[VacancyPreview],
        profile: Profile,
        preferences: list[PreferenceIntent],
        query: VacancyScrapingQuery,
        *,
        batch_size: int,
    ) -> list[VacancyPreview]:
        if batch_size < 1:
            raise ValueError("batch_size must be greater than zero")
        accepted: list[VacancyPreview] = []
        for offset in range(0, len(previews), batch_size):
            batch = previews[offset : offset + batch_size]
            states = []
            for preview in batch:
                matching = [
                    preference
                    for preference in preferences
                    if preference.enabled
                    and self._matches_preference(preview, preference)
                ]
                states.append(
                    {
                        "candidate": {
                            "target_role": "; ".join(
                                map(build_preference_title_text, matching),
                            )
                            or query.text,
                            "skills": profile.skills,
                            "experience": profile.experience,
                            "seniority": profile.seniority,
                            "experience_years": profile.experience_years,
                            "preferences": [
                                {
                                    "description": preference.description,
                                    "required_skills": preference.required_skills,
                                    "preferred_skills": preference.preferred_skills,
                                    "locations": preference.locations,
                                    "work_formats": preference.work_formats,
                                    "employment_types": preference.employment_types,
                                    "salary_min": preference.salary_min,
                                    "salary_currency": preference.salary_currency,
                                    "min_seniority": preference.min_seniority,
                                    "max_seniority": preference.max_seniority,
                                }
                                for preference in matching
                            ],
                        },
                        "vacancy": preview.model_dump(mode="json"),
                    },
                )
            try:
                results = await self._laya.evaluate_batch(
                    states,
                    MATCH_QUESTIONS,
                    batch_size=batch_size,
                )
                if len(results) != len(batch):
                    raise ValueError(
                        "Laya returned a different number of preview results"
                    )
                comparisons = [
                    LayaComparison(
                        role_fit=result["answers"]["role_fit"]["choice"],
                        skill_fit=result["answers"]["skill_fit"]["choice"],
                    )
                    for result in results
                ]
            except Exception as error:
                raise VacancyPreviewEvaluationError(
                    "Vacancy preview evaluation failed",
                ) from error
            for preview, comparison in zip(batch, comparisons, strict=True):
                # ponytail: ordinal preview cutoffs are uncalibrated; tune against labeled previews.
                if (
                    comparison.role_fit in {"good", "strong"}
                    and comparison.skill_fit != "none"
                ):
                    accepted.append(preview)
        return accepted
