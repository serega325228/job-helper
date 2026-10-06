import math
from typing import ClassVar
from uuid import UUID

from src.exceptions.profile import ProfileNotFoundError
from src.infrastructure.db.sqlalchemy_unit_of_work import SqlAlchemyUnitOfWork
from src.infrastructure.models.vacancy_match import VacancyMatch
from src.schemas.scoring import LayaComparison, PreferenceComparison, ProfileComparison
from src.schemas.vacancy import (
    BatchStatus,
    ProcessingStatus,
    RawVacancy,
    VacancyHardFilters,
)
from src.schemas.vacancy_match import (
    MatchCategory,
    VacancyEmbeddingSearchResult,
    VacancyMatchResult,
)
from src.services.scoring import ScoringService
from src.services.vacancy import VacancyService


class VacancyMatchService:
    # ponytail: ordinal fit scores are uncalibrated; calibrate against labeled matches.
    LAYA_FIT_SCORES: ClassVar[dict[str, float]] = {
        "none": 0.0,
        "weak": 1 / 3,
        "good": 2 / 3,
        "strong": 1.0,
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

    async def score_embeddings(self, batch_id: UUID, service: VacancyService) -> bool:
        async with self._uow as uow:
            batch = await uow.vacancies.get_batch(batch_id)
            if batch is None or batch.status != BatchStatus.DETAILS:
                return False
            vacancies = await uow.vacancies.get_batch_vacancies(batch_id)
            if any(
                vacancy.processing_status
                in {ProcessingStatus.PENDING_SCRAPE, ProcessingStatus.PENDING_PARSE}
                for vacancy in vacancies
            ):
                return False
            pending = [
                vacancy
                for vacancy in vacancies
                if vacancy.processing_status == ProcessingStatus.PENDING_EMBEDDING
            ]
            if not pending:
                batch.status = BatchStatus.COMPLETED
                return False
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
            await self._scoring.update_preference_embeddings(
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
            results = await uow.vacancy_matches.search_by_preferences(
                batch.profile_id,
                VacancyHardFilters.model_validate(batch.hard_filters),
                limit=batch.search_limit,
                title_weight=batch.title_weight,
                candidate_limit=len(pending),
                per_preference_limit=len(pending),
                vacancy_ids=[vacancy.id for vacancy in pending],
                preference_ids=[preference.id for preference in preferences],
            )
            existing = {
                match.vacancy_id: match
                for match in await uow.vacancy_matches.get_by_profile_and_vacancy_ids(
                    batch.profile_id, [result.vacancy_id for result in results]
                )
            }
            by_id = {vacancy.id: vacancy for vacancy in pending}
            preferences_by_id = {
                preference.id: preference for preference in preferences
            }
            selected_ids = set()
            new_matches = []
            for result in results:
                vacancy = by_id[result.vacancy_id]
                preference = preferences_by_id[result.preference_id]
                profile_comparison = self._scoring.compare_profile(profile, vacancy)
                preference_comparison = self._scoring.compare_preference(
                    preference, vacancy
                )
                values = {
                    "profile_id": profile.id,
                    "vacancy_id": vacancy.id,
                    "preference_intent_id": preference.id,
                    "structured_profile_score": profile_comparison.score,
                    "structured_preference_score": preference_comparison.score,
                    "profile_rerank_score": None,
                    "preference_rerank_score": None,
                    "total_score": None,
                    "category": None,
                    "hard_constraints_passed": preference_comparison.hard_constraints_passed,
                    "component_scores": {
                        "profile": profile_comparison.components,
                        "preference": preference_comparison.components,
                        "semantic": {
                            "title": result.title_similarity,
                            "content": result.content_similarity,
                            "combined": result.combined_similarity,
                        },
                    },
                    "matched_skills": profile_comparison.matched_skills,
                    "missing_skills": profile_comparison.missing_skills,
                    "matcher_version": "structured-laya-rerank-v1",
                }
                match = existing.get(vacancy.id)
                if match is None:
                    new_matches.append(VacancyMatch(**values))
                else:
                    for field, value in values.items():
                        setattr(match, field, value)
                if (
                    preference_comparison.hard_constraints_passed
                    and len(selected_ids) < batch.rerank_limit
                ):
                    selected_ids.add(vacancy.id)
            uow.vacancy_matches.add_all(new_matches)
            for vacancy in pending:
                vacancy.processing_status = (
                    ProcessingStatus.PENDING_RERANK
                    if vacancy.id in selected_ids
                    else ProcessingStatus.FILTERED
                )
            batch.status = (
                BatchStatus.PENDING_RERANK if selected_ids else BatchStatus.COMPLETED
            )
        return bool(selected_ids)

    async def rerank_batch(self, batch_id: UUID) -> bool:
        async with self._uow as uow:
            batch = await uow.vacancies.get_batch(batch_id)
            if batch is None or batch.status != BatchStatus.PENDING_RERANK:
                return False
            vacancies = [
                vacancy
                for vacancy in await uow.vacancies.get_batch_vacancies(batch_id)
                if vacancy.processing_status == ProcessingStatus.PENDING_RERANK
            ]
            if not vacancies:
                return False
            profile = await uow.profiles.get_by_id(batch.profile_id)
            if profile is None:
                raise ProfileNotFoundError(batch.profile_id)
            matches = {
                match.vacancy_id: match
                for match in await uow.vacancy_matches.get_by_profile_and_vacancy_ids(
                    profile.id, [vacancy.id for vacancy in vacancies]
                )
            }
            preferences = {
                preference.id: preference
                for preference in await uow.profiles.get_preferences_by_ids(
                    profile.id,
                    [match.preference_intent_id for match in matches.values()],
                )
            }
            scores = await self._scoring.rerank_vacancies(
                profile,
                vacancies,
                {
                    vacancy.id: preferences[matches[vacancy.id].preference_intent_id]
                    for vacancy in vacancies
                },
            )
            for vacancy in vacancies:
                match = matches[vacancy.id]
                match.profile_rerank_score = scores[vacancy.id].profile_score
                match.preference_rerank_score = scores[vacancy.id].preference_score
                vacancy.processing_status = ProcessingStatus.PENDING_LAYA
            batch.status = BatchStatus.PENDING_LAYA
        return True

    async def evaluate_batch(self, batch_id: UUID, *, batch_size: int) -> bool:
        async with self._uow as uow:
            batch = await uow.vacancies.get_batch(batch_id)
            if batch is None or batch.status != BatchStatus.PENDING_LAYA:
                return False
            vacancies = [
                vacancy
                for vacancy in await uow.vacancies.get_batch_vacancies(batch_id)
                if vacancy.processing_status == ProcessingStatus.PENDING_LAYA
            ]
            if not vacancies:
                return False
            profile = await uow.profiles.get_by_id(batch.profile_id)
            if profile is None:
                raise ProfileNotFoundError(batch.profile_id)
            matches = {
                match.vacancy_id: match
                for match in await uow.vacancy_matches.get_by_profile_and_vacancy_ids(
                    profile.id, [vacancy.id for vacancy in vacancies]
                )
            }
            if any(
                matches[vacancy.id].profile_rerank_score is None
                or matches[vacancy.id].preference_rerank_score is None
                for vacancy in vacancies
            ):
                raise ValueError("Detailed Laya evaluation requires rerank scores")
            preferences = {
                preference.id: preference
                for preference in await uow.profiles.get_preferences_by_ids(
                    profile.id,
                    [match.preference_intent_id for match in matches.values()],
                )
            }
            evaluations = await self._scoring.laya_match_vacancies(
                profile,
                {
                    vacancy.id: preferences[matches[vacancy.id].preference_intent_id]
                    for vacancy in vacancies
                },
                {vacancy.id: vacancy for vacancy in vacancies},
                batch_size=batch_size,
            )
            for vacancy in vacancies:
                match = matches[vacancy.id]
                match.component_scores = {
                    **match.component_scores,
                    "laya": evaluations[vacancy.id].model_dump(mode="json"),
                }
                vacancy.processing_status = ProcessingStatus.PENDING_SAVE
            batch.status = BatchStatus.PENDING_SAVE
        return True

    async def save_batch_scores(self, batch_id: UUID) -> None:
        async with self._uow as uow:
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
            matches = {
                match.vacancy_id: match
                for match in await uow.vacancy_matches.get_by_profile_and_vacancy_ids(
                    batch.profile_id, [vacancy.id for vacancy in vacancies]
                )
            }
            for vacancy in vacancies:
                match = matches[vacancy.id]
                semantic = match.component_scores["semantic"]
                result = self.build_result(
                    profile_id=match.profile_id,
                    vacancy_id=match.vacancy_id,
                    preference_intent_id=match.preference_intent_id,
                    profile_comparison=ProfileComparison(
                        score=match.structured_profile_score,
                        components=match.component_scores["profile"],
                        matched_skills=match.matched_skills,
                        missing_skills=match.missing_skills,
                    ),
                    preference_comparison=PreferenceComparison(
                        score=match.structured_preference_score,
                        components=match.component_scores["preference"],
                        hard_constraints_passed=match.hard_constraints_passed,
                    ),
                    laya_comparison=LayaComparison.model_validate(
                        match.component_scores["laya"]
                    ),
                    profile_rerank_score=match.profile_rerank_score,
                    preference_rerank_score=match.preference_rerank_score,
                    title_similarity=semantic["title"],
                    content_similarity=semantic["content"],
                    embedding_similarity=semantic["combined"],
                )
                values = result.model_dump(mode="python")
                values["category"] = result.category.value
                for field, value in values.items():
                    setattr(match, field, value)
                vacancy.processing_status = ProcessingStatus.COMPLETED
            batch.status = BatchStatus.COMPLETED

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
