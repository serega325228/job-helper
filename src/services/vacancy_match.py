import math
from typing import ClassVar
from uuid import UUID

from src.infrastructure.db.sqlalchemy_unit_of_work import SqlAlchemyUnitOfWork
from src.infrastructure.models.vacancy_match import VacancyMatch
from src.schemas.scoring import PreferenceComparison, ProfileComparison
from src.schemas.vacancy import VacancyHardFilters
from src.schemas.vacancy_match import (
    MatchCategory,
    VacancyEmbeddingSearchResult,
    VacancyMatchResult,
)


class VacancyMatchService:
    FINAL_SCORE_WEIGHTS: ClassVar[dict[str, float]] = {
        "structured_profile": 0.20,
        "structured_preference": 0.20,
        "profile_rerank": 0.30,
        "preference_rerank": 0.30,
    }

    def __init__(self, unit_of_work: SqlAlchemyUnitOfWork) -> None:
        self._uow = unit_of_work

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

        scores = {
            "structured_profile": profile_comparison.score,
            "structured_preference": preference_comparison.score,
            "profile_rerank": profile_rerank_score,
            "preference_rerank": preference_rerank_score,
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
    def geometric_score(
        first: float,
        second: float,
        *,
        first_weight: float,
    ) -> float:
        return VacancyMatchService.weighted_geometric_score(
            {"first": first, "second": second},
            {"first": first_weight, "second": 1 - first_weight},
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
