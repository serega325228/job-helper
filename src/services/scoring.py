import asyncio
import logging
from datetime import UTC, datetime

from infrastructure.models.preference_intent import PreferenceIntent
from infrastructure.models.profile import Profile
from infrastructure.reranker.vacancy_reranker import VacancyReranker
from pydantic import ValidationError
from schemas.vacancy_match import VacancyMatchResult

from src.config.retry import RetryableLlmError, llm_retry
from src.exceptions.vacancy import VacancyNormalizationError
from src.infrastructure.models.vacancy import Vacancy
from src.ports.vacancy_source import VacancySource
from src.schemas.vacancy import (
    NormalizedVacancy,
    RawVacancy,
    VacancyReference,
    VacancySearchQuery,
)

logger = logging.getLogger(__name__)


class ScoringService:
    def __init__(
        self,
        reranker: VacancyReranker,
        rerank_limit: int = 40,
    ) -> None:
        self._reranker = reranker
        self._rerank_limit = rerank_limit

    def _build_preference_query(self, preference: PreferenceIntent) -> str:
        parts = [
            f"Target position: {preference.target_title}.",
            f"Experience: {preference.experience_years:g} years.",
        ]

        if preference.seniority:
            parts.append(
                f"Target seniority: {preference.seniority.value}."
            )

        if preference.skills:
            parts.append(
                "Skills: " + ", ".join(preference.skills) + "."
            )

        if preference.desired_work_formats:
            formats = ", ".join(
                item.value
                for item in preference.desired_work_formats
            )

            parts.append(f"Preferred work format: {formats}.")

        return "\n".join(parts)

    def _build_vacancy_document(
        self,
        vacancy: NormalizedVacancy,
    ) -> str:
        parts = [
            f"Position: {vacancy.title}."
        ]

        if vacancy.seniority:
            parts.append(
                f"Seniority: {vacancy.seniority.value}."
            )

        if vacancy.min_experience_years is not None:
            parts.append(
                "Minimum experience: "
                f"{vacancy.min_experience_years:g} years."
            )

        if vacancy.required_skills:
            parts.append(
                "Required skills: "
                + ", ".join(vacancy.required_skills)
                + "."
            )

        if vacancy.preferred_skills:
            parts.append(
                "Preferred skills: "
                + ", ".join(vacancy.preferred_skills)
                + "."
            )

        if vacancy.responsibilities:
            parts.append(
                "Responsibilities: "
                + "; ".join(vacancy.responsibilities)
                + "."
            )

        if vacancy.work_format:
            parts.append(
                f"Work format: {vacancy.work_format.value}."
            )

        return "\n".join(parts)

    def _normalize_skill(self, skill: str) -> str:
        return skill.strip().lower()

    async def score(
        self,
        vacancies: list[NormalizedVacancy],
        profile: Profile,
        preferences: list[PreferenceIntent],
    ) -> list[ScoredVacancy]:
        if not vacancies:
            return []

        intent_match = self._preference_scorer.score(
            vacancy,
            preferences.intents,
        )

        profile_match = self._profile_scorer.score(
            vacancy,
            profile,
        )

        category = classify_match(
            profile_fit=profile_match.score,
            preference_fit=intent_match.score,
        )

        final_score = (
            profile_match.score ** 0.4
            * intent_match.score ** 0.6
        )

        return VacancyMatchResult(
            vacancy_id=vacancy.id,
            profile_fit=profile_match.score,
            preference_fit=intent_match.score,
            matched_intent_id=intent_match.intent_id,
            final_score=final_score,
            category=category,
            missing_required_skills=profile_match.missing_skills,
            matched_skills=profile_match.matched_skills,
        )

    def _profile_scorer(self, profile: Profile) -> ...:
       return ...

    def _preference_scorer(self, profile: Profile) -> ...:
        return ...
