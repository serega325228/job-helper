import math
from typing import ClassVar
from uuid import UUID

from src.infrastructure.db.sqlalchemy_unit_of_work import SqlAlchemyUnitOfWork
from src.infrastructure.models.preference_intent import PreferenceIntent
from src.infrastructure.models.profile import Profile
from src.infrastructure.models.vacancy import Vacancy
from src.infrastructure.models.vacancy_match import VacancyMatch
from src.schemas.scoring import LayaComparison, PreferenceComparison, ProfileComparison
from src.schemas.vacancy import VacancyHardFilters
from src.schemas.vacancy_match import (
    MatchCategory,
    MatchingCandidate,
    VacancyEmbeddingSearchResult,
    VacancyMatchResult,
)
from src.services.scoring import ScoringService


class VacancyMatchService:
    # ponytail: ordinal fit scores are uncalibrated; calibrate against labeled matches.
    LAYA_FIT_SCORES: ClassVar[dict[str, float]] = {
        "none": 0.0,
        "weak": 1 / 3,
        "good": 2 / 3,
        "strong": 1.0,
    }
    COMPARISON_SCORE_WEIGHTS: ClassVar[dict[str, float]] = {
        "structured_profile": 0.36,
        "structured_preference": 0.44,
        "laya_role": 0.10,
        "laya_skill": 0.10,
    }
    FINAL_SCORE_WEIGHTS: ClassVar[dict[str, float]] = {
        "structured_profile": 0.16,
        "structured_preference": 0.16,
        "profile_rerank": 0.24,
        "preference_rerank": 0.24,
        "laya_role": 0.10,
        "laya_skill": 0.10,
    }

    def __init__(
        self,
        unit_of_work: SqlAlchemyUnitOfWork,
        scoring_service: ScoringService,
    ) -> None:
        self._uow = unit_of_work
        self._scoring = scoring_service

    async def compare_candidates(
        self,
        profile: Profile,
        candidates: dict[UUID, MatchingCandidate],
        vacancies: dict[UUID, Vacancy],
        preferences: dict[UUID, PreferenceIntent],
        *,
        limit: int,
    ) -> dict[UUID, MatchingCandidate]:
        if limit < 1:
            raise ValueError("limit must be greater than zero")

        compared: dict[UUID, MatchingCandidate] = {}
        for vacancy_id, candidate in candidates.items():
            vacancy = vacancies[vacancy_id]
            preference_comparison = self._scoring.compare_preference(
                preferences[candidate.preference_id],
                vacancy,
            )
            if not preference_comparison.hard_constraints_passed:
                continue

            profile_comparison = self._scoring.compare_profile(profile, vacancy)
            compared[vacancy_id] = candidate.model_copy(
                update={
                    "profile_comparison": profile_comparison,
                    "preference_comparison": preference_comparison,
                },
            )

        if not compared:
            return {}

        preferences_by_vacancy = {
            vacancy_id: preferences[candidate.preference_id]
            for vacancy_id, candidate in compared.items()
        }
        laya_matches = await self._scoring.laya_match_vacancies(
            profile,
            preferences_by_vacancy,
            {vacancy_id: vacancies[vacancy_id] for vacancy_id in compared},
        )
        for vacancy_id, candidate in compared.items():
            laya_comparison = laya_matches[vacancy_id]
            candidate.laya_comparison = laya_comparison
            candidate.structured_score = self.weighted_geometric_score(
                {
                    "structured_profile": candidate.profile_comparison.score,
                    "structured_preference": candidate.preference_comparison.score,
                    "laya_role": self.LAYA_FIT_SCORES[laya_comparison.role_fit],
                    "laya_skill": self.LAYA_FIT_SCORES[laya_comparison.skill_fit],
                },
                self.COMPARISON_SCORE_WEIGHTS,
            )

        ranked = dict(
            sorted(
                compared.items(),
                key=lambda item: item[1].structured_score,
                reverse=True,
            )[:limit],
        )
        rerank_scores = await self._scoring.rerank_vacancies(
            profile,
            [vacancies[vacancy_id] for vacancy_id in ranked],
            {vacancy_id: preferences_by_vacancy[vacancy_id] for vacancy_id in ranked},
        )
        for vacancy_id, candidate in ranked.items():
            candidate.profile_rerank_score = rerank_scores[vacancy_id].profile_score
            candidate.preference_rerank_score = (
                rerank_scores[vacancy_id].preference_score
            )
        return ranked

    async def save_result(self, result: VacancyMatchResult) -> VacancyMatch:
        return (await self.save_results([result]))[0]

    async def save_results(
        self,
        results: list[VacancyMatchResult],
    ) -> list[VacancyMatch]:
        if not results:
            return []

        profile_ids = {result.profile_id for result in results}
        if len(profile_ids) != 1:
            raise ValueError("All vacancy matches must belong to the same profile")
        vacancy_ids = [result.vacancy_id for result in results]
        if len(vacancy_ids) != len(set(vacancy_ids)):
            raise ValueError("Vacancy match results contain duplicate vacancies")

        async with self._uow as uow:
            existing_matches = {
                vacancy_match.vacancy_id: vacancy_match
                for vacancy_match in (
                    await uow.vacancy_matches.get_by_profile_and_vacancy_ids(
                        next(iter(profile_ids)),
                        vacancy_ids,
                    )
                )
            }
            saved_matches: list[VacancyMatch] = []
            new_matches: list[VacancyMatch] = []
            for result in results:
                values = result.model_dump(mode="python")
                values["category"] = result.category.value
                vacancy_match = existing_matches.get(result.vacancy_id)
                if vacancy_match is None:
                    vacancy_match = VacancyMatch(**values)
                    new_matches.append(vacancy_match)
                else:
                    for field, value in values.items():
                        setattr(vacancy_match, field, value)
                saved_matches.append(vacancy_match)

            uow.vacancy_matches.add_all(new_matches)
            await uow.flush()

        return saved_matches

    def build_result(
        self,
        *,
        profile_id: UUID,
        vacancy_id: UUID,
        preference_intent_id: UUID,
        profile_comparison: ProfileComparison | None,
        preference_comparison: PreferenceComparison | None,
        laya_comparison: LayaComparison | None,
        profile_rerank_score: float | None,
        preference_rerank_score: float | None,
        title_similarity: float,
        content_similarity: float,
        embedding_similarity: float,
    ) -> VacancyMatchResult:
        if profile_comparison is None or preference_comparison is None:
            raise ValueError(f"Structured scores are missing for vacancy {vacancy_id}")
        if profile_rerank_score is None or preference_rerank_score is None:
            raise ValueError(f"Rerank scores are missing for vacancy {vacancy_id}")
        if laya_comparison is None:
            raise ValueError(f"Laya scores are missing for vacancy {vacancy_id}")

        scores = {
            "structured_profile": profile_comparison.score,
            "structured_preference": preference_comparison.score,
            "profile_rerank": profile_rerank_score,
            "preference_rerank": preference_rerank_score,
            "laya_role": self.LAYA_FIT_SCORES[laya_comparison.role_fit],
            "laya_skill": self.LAYA_FIT_SCORES[laya_comparison.skill_fit],
        }
        total_score = (
            self.weighted_geometric_score(scores, self.FINAL_SCORE_WEIGHTS)
            if preference_comparison.hard_constraints_passed
            else 0.0
        )
        return VacancyMatchResult(
            profile_id=profile_id,
            vacancy_id=vacancy_id,
            preference_intent_id=preference_intent_id,
            structured_profile_score=profile_comparison.score,
            structured_preference_score=preference_comparison.score,
            profile_rerank_score=profile_rerank_score,
            preference_rerank_score=preference_rerank_score,
            total_score=total_score,
            category=self._category_for_score(total_score),
            hard_constraints_passed=preference_comparison.hard_constraints_passed,
            component_scores={
                "profile": profile_comparison.components,
                "preference": preference_comparison.components,
                "laya": {
                    **laya_comparison.model_dump(),
                    "role_score": scores["laya_role"],
                    "skill_score": scores["laya_skill"],
                },
                "semantic": {
                    "title": title_similarity,
                    "content": content_similarity,
                    "combined": embedding_similarity,
                },
            },
            matched_skills=profile_comparison.matched_skills,
            missing_skills=profile_comparison.missing_skills,
        )

    @staticmethod
    def weighted_geometric_score(
        scores: dict[str, float],
        weights: dict[str, float],
    ) -> float:
        if scores.keys() != weights.keys():
            raise ValueError("Scores and weights must have the same keys")
        total_weight = sum(weights.values())
        if total_weight <= 0 or any(weight < 0 for weight in weights.values()):
            raise ValueError("Weights must be non-negative and have a positive sum")
        if any(not 0 <= score <= 1 for score in scores.values()):
            raise ValueError("Scores must be between zero and one")
        if any(score == 0 for score in scores.values()):
            return 0.0
        return math.prod(
            score ** (weights[name] / total_weight) for name, score in scores.items()
        )

    @staticmethod
    def _category_for_score(score: float) -> MatchCategory:
        if score >= 0.80:
            return MatchCategory.TARGET
        if score >= 0.65:
            return MatchCategory.STRETCH
        if score >= 0.45:
            return MatchCategory.FALLBACK
        return MatchCategory.REJECT

    async def search_by_preferences(
        self,
        profile_id: UUID,
        hard_filters: VacancyHardFilters,
        *,
        limit: int = 100,
        title_weight: float = 0.4,
        candidate_limit: int = 100,
        per_preference_limit: int = 50,
    ) -> list[VacancyEmbeddingSearchResult]:
        async with self._uow as uow:
            return await uow.vacancy_matches.search_by_preferences(
                profile_id,
                hard_filters,
                limit=limit,
                title_weight=title_weight,
                candidate_limit=candidate_limit,
                per_preference_limit=per_preference_limit,
            )
