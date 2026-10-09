import math
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import ClassVar
from uuid import UUID, uuid4

from src.config.settings import Settings
from src.exceptions.profile import ProfileNotFoundError
from src.infrastructure.db.sqlalchemy_unit_of_work import SqlAlchemyUnitOfWork
from src.schemas.scoring import LayaComparison, PreferenceComparison, ProfileComparison
from src.schemas.vacancy import BatchStatus, ProcessingStatus, VacancyHardFilters
from src.schemas.vacancy_match import MatchCategory, VacancyMatchResult
from src.services.scoring import ScoringService


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
        settings: Settings,
    ) -> None:
        self._uow = unit_of_work
        self._scoring = scoring_service
        self._settings = settings

    @asynccontextmanager
    async def _stage(self, batch_id: UUID, expected: BatchStatus):
        token = uuid4()
        async with self._uow as uow:
            batch = await uow.vacancies.get_batch(batch_id)
            claimed = False
            if batch is not None and batch.status == expected:
                cutoff = datetime.now(UTC) - timedelta(
                    seconds=self._settings.taskiq.lease_timeout_seconds
                )
                if (
                    batch.processing_token is not None
                    and batch.updated_at >= cutoff
                    or expected == BatchStatus.DETAILS
                    and any(
                        vacancy.processing_status
                        in {
                            ProcessingStatus.PENDING_SCRAPE,
                            ProcessingStatus.PENDING_PARSE,
                        }
                        for vacancy in await uow.vacancies.get_vacancies_by_ids(
                            [UUID(value) for value in batch.vacancy_ids]
                        )
                    )
                ):
                    batch = None
                elif batch.attempts >= self._settings.taskiq.stage_attempts:
                    batch.status = BatchStatus.FAILED
                    batch.error = "Stage attempts exhausted"
                    batch.processing_token = None
                    await uow.vacancy_matches.fail_batch(batch_id)
                else:
                    batch.processing_token = token
                    batch.attempts += 1
                    batch.updated_at = datetime.now(UTC)
                    claimed = True
        try:
            yield batch, token
        except Exception as error:
            if claimed:
                async with self._uow as uow:
                    row = await uow.vacancies.get_batch(batch_id)
                    if row.processing_token == token:
                        row.processing_token = None
                        row.error = str(error)
                        if (
                            isinstance(error, (ValueError, ProfileNotFoundError))
                            or row.attempts >= self._settings.taskiq.stage_attempts
                        ):
                            row.status = BatchStatus.FAILED
                            await uow.vacancy_matches.fail_batch(batch_id)
            raise

    async def _inputs(self, batch):
        async with self._uow as uow:
            profile = await uow.profiles.get_by_id(batch.profile_id)
            if profile is None:
                raise ProfileNotFoundError(batch.profile_id)
            preferences = await uow.profiles.get_preferences_by_ids(
                batch.profile_id,
                [UUID(value) for value in batch.preference_ids],
                enabled_only=True,
            )
            if not preferences:
                raise ValueError("No enabled preferences remain for this batch")
            vacancies = await uow.vacancies.get_vacancies_by_ids(
                [UUID(value) for value in batch.vacancy_ids]
            )
            matches = await uow.vacancy_matches.get_batch_matches(batch.id)
        if len(vacancies) != len(batch.vacancy_ids):
            raise ValueError("Batch vacancies are missing")
        preferences_by_id = {item.id: item for item in preferences}
        if batch.status != BatchStatus.DETAILS and any(
            match.preference_intent_id not in preferences_by_id
            for match in matches
            if match.processing_status
            not in {ProcessingStatus.FILTERED, ProcessingStatus.COMPLETED}
        ):
            raise ValueError("A shortlisted preference was disabled or removed")
        vacancies_by_id = {item.id: item for item in vacancies}
        if batch.status != BatchStatus.DETAILS and any(
            match.processing_status
            not in {ProcessingStatus.COMPLETED, ProcessingStatus.FILTERED}
            and vacancies_by_id[match.vacancy_id].processing_status
            != ProcessingStatus.COMPLETED
            for match in matches
        ):
            raise ValueError("A shortlisted vacancy is not fully parsed")
        return profile, preferences_by_id, vacancies_by_id, matches

    @staticmethod
    def _finish(batch, status: BatchStatus) -> None:
        batch.status = status
        batch.processing_token = None
        batch.attempts = 0
        batch.error = None

    async def score_embeddings(self, batch_id: UUID) -> bool:
        async with self._stage(batch_id, BatchStatus.DETAILS) as (batch, token):
            if batch is None or batch.status != BatchStatus.DETAILS:
                return batch is not None and batch.status == BatchStatus.PENDING_RERANK
            profile, preferences, vacancies, _ = await self._inputs(batch)
            eligible = {
                identifier: vacancy
                for identifier, vacancy in vacancies.items()
                if vacancy.processing_status == ProcessingStatus.COMPLETED
            }
            if not eligible:
                async with self._uow as uow:
                    row = await uow.vacancies.get_batch(batch_id)
                    if row.processing_token == token:
                        failed = any(
                            item.processing_status == ProcessingStatus.FAILED
                            for item in vacancies.values()
                        )
                        self._finish(
                            row, BatchStatus.FAILED if failed else BatchStatus.COMPLETED
                        )
                        if failed:
                            row.error = "All available vacancy details failed"
                return False
            entities = [*preferences.values(), *eligible.values()]
            fingerprints = {
                entity.id: self._scoring.embedding_fingerprint(entity)
                for entity in entities
            }
            embeddings = await self._scoring.embed_entities(entities)
            async with self._uow as uow:
                row = await uow.vacancies.get_batch(batch_id)
                if row.processing_token != token:
                    return False
                current_preferences = await uow.profiles.get_preferences_by_ids(
                    batch.profile_id, list(preferences), enabled_only=True
                )
                current_vacancies = await uow.vacancies.get_vacancies_by_ids(
                    list(eligible), lock=True
                )
                for entity in [*current_preferences, *current_vacancies]:
                    if (
                        self._scoring.embedding_fingerprint(entity)
                        != fingerprints[entity.id]
                    ):
                        raise RuntimeError("Embedding input changed while processing")
                    if entity.id in embeddings:
                        title, content, fingerprint = embeddings[entity.id]
                        entity.title_embedding = title
                        entity.content_embedding = content
                        entity.embedding_fingerprint = fingerprint
                await uow.flush()
                results = await uow.vacancy_matches.embedding_scores(
                    batch.profile_id,
                    VacancyHardFilters.model_validate(batch.hard_filters),
                    list(eligible),
                    list(preferences),
                    batch.title_weight,
                    selection_context=batch.selection_context,
                )
                best = {}
                for result in results:
                    vacancy = eligible[result.vacancy_id]
                    preference = preferences[result.preference_id]
                    comparison = self._scoring.compare_preference(preference, vacancy)
                    if not comparison.hard_constraints_passed:
                        continue
                    selection_score = result.combined_similarity * preference.weight
                    previous = best.get(vacancy.id)
                    if previous is None or selection_score > previous[0]:
                        best[vacancy.id] = (selection_score, result, comparison)
                ordered = sorted(
                    best.values(), key=lambda item: (-item[0], str(item[1].vacancy_id))
                )
                shortlisted = {
                    item[1].vacancy_id
                    for item in ordered[: min(batch.search_limit, batch.rerank_limit)]
                }
                values = []
                for selection_score, result, comparison in ordered:
                    vacancy = eligible[result.vacancy_id]
                    profile_comparison = self._scoring.compare_profile(profile, vacancy)
                    values.append(
                        {
                            "profile_id": profile.id,
                            "vacancy_id": vacancy.id,
                            "batch_id": batch_id,
                            "preference_intent_id": result.preference_id,
                            "processing_status": ProcessingStatus.PENDING_RERANK
                            if vacancy.id in shortlisted
                            else ProcessingStatus.FILTERED,
                            "structured_profile_score": profile_comparison.score,
                            "structured_preference_score": comparison.score,
                            "profile_rerank_score": None,
                            "preference_rerank_score": None,
                            "total_score": None,
                            "category": None,
                            "hard_constraints_passed": True,
                            "component_scores": {
                                "profile": profile_comparison.components,
                                "preference": comparison.components,
                                "semantic": {
                                    "title": result.title_similarity,
                                    "content": result.content_similarity,
                                    "combined": result.combined_similarity,
                                    "selection": selection_score,
                                },
                            },
                            "matched_skills": profile_comparison.matched_skills,
                            "missing_skills": profile_comparison.missing_skills,
                            "matcher_version": "structured-laya-rerank-v1",
                            "explanation": None,
                            "reranker_model": None,
                        }
                    )
                for match in await uow.vacancy_matches.get_by_profile_and_vacancy_ids(
                    batch.profile_id,
                    [identifier for identifier in vacancies if identifier not in best],
                ):
                    match.batch_id = batch_id
                    match.processing_status = ProcessingStatus.FILTERED
                    match.total_score = None
                    match.category = None
                await uow.vacancy_matches.upsert_initial(values)
                self._finish(
                    row,
                    BatchStatus.PENDING_RERANK
                    if shortlisted
                    else BatchStatus.COMPLETED,
                )
            return bool(shortlisted)

    async def rerank_batch(self, batch_id: UUID) -> bool:
        async with self._stage(batch_id, BatchStatus.PENDING_RERANK) as (batch, token):
            if batch is None or batch.status != BatchStatus.PENDING_RERANK:
                return batch is not None and batch.status == BatchStatus.PENDING_LAYA
            profile, preferences, vacancies, matches = await self._inputs(batch)
            pending = [
                match
                for match in matches
                if match.processing_status == ProcessingStatus.PENDING_RERANK
            ]
            if not pending:
                raise ValueError("Rerank shortlist is missing")
            scores = await self._scoring.rerank_vacancies(
                profile,
                [vacancies[match.vacancy_id] for match in pending],
                {
                    match.vacancy_id: preferences[match.preference_intent_id]
                    for match in pending
                },
            )
            async with self._uow as uow:
                row = await uow.vacancies.get_batch(batch_id)
                if row.processing_token != token:
                    return False
                for match in await uow.vacancy_matches.get_batch_matches(
                    batch_id, lock=True
                ):
                    if match.processing_status != ProcessingStatus.PENDING_RERANK:
                        continue
                    match.profile_rerank_score = scores[match.vacancy_id].profile_score
                    match.preference_rerank_score = scores[
                        match.vacancy_id
                    ].preference_score
                    match.processing_status = ProcessingStatus.PENDING_LAYA
                self._finish(row, BatchStatus.PENDING_LAYA)
            return True

    async def evaluate_batch(self, batch_id: UUID, *, batch_size: int) -> bool:
        async with self._stage(batch_id, BatchStatus.PENDING_LAYA) as (batch, token):
            if batch is None or batch.status != BatchStatus.PENDING_LAYA:
                return batch is not None and batch.status == BatchStatus.PENDING_SAVE
            profile, preferences, vacancies, matches = await self._inputs(batch)
            pending = [
                match
                for match in matches
                if match.processing_status == ProcessingStatus.PENDING_LAYA
            ]
            if not any(
                match.processing_status
                in {ProcessingStatus.PENDING_LAYA, ProcessingStatus.PENDING_SAVE}
                for match in matches
            ):
                raise ValueError("Detailed evaluation shortlist is missing")
            if any(
                match.profile_rerank_score is None
                or match.preference_rerank_score is None
                for match in pending
            ):
                raise ValueError("Detailed Laya evaluation requires rerank scores")
            for offset in range(0, len(pending), batch_size):
                chunk = pending[offset : offset + batch_size]
                evaluations = await self._scoring.laya_match_vacancies(
                    profile,
                    {
                        match.vacancy_id: preferences[match.preference_intent_id]
                        for match in chunk
                    },
                    {match.vacancy_id: vacancies[match.vacancy_id] for match in chunk},
                    batch_size=batch_size,
                )
                async with self._uow as uow:
                    row = await uow.vacancies.get_batch(batch_id)
                    if row.processing_token != token:
                        return False
                    for match in await uow.vacancy_matches.get_batch_matches(
                        batch_id, lock=True
                    ):
                        if (
                            match.processing_status == ProcessingStatus.PENDING_LAYA
                            and match.vacancy_id in evaluations
                        ):
                            match.component_scores = {
                                **match.component_scores,
                                "laya": evaluations[match.vacancy_id].model_dump(
                                    mode="json"
                                ),
                            }
                            match.processing_status = ProcessingStatus.PENDING_SAVE
                    row.updated_at = datetime.now(UTC)
            async with self._uow as uow:
                row = await uow.vacancies.get_batch(batch_id)
                if row.processing_token != token:
                    return False
                self._finish(row, BatchStatus.PENDING_SAVE)
            return True

    async def save_batch_scores(self, batch_id: UUID) -> None:
        async with self._stage(batch_id, BatchStatus.PENDING_SAVE) as (batch, token):
            if batch is None or batch.status != BatchStatus.PENDING_SAVE:
                return
            async with self._uow as uow:
                row = await uow.vacancies.get_batch(batch_id)
                if row.processing_token != token:
                    return
                matches = await uow.vacancy_matches.get_batch_matches(
                    batch_id, lock=True
                )
                if not any(
                    match.processing_status == ProcessingStatus.PENDING_SAVE
                    for match in matches
                ):
                    raise ValueError("Detailed evaluation results are missing")
                for match in matches:
                    if match.processing_status != ProcessingStatus.PENDING_SAVE:
                        continue
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
                    values["component_scores"]["semantic"] = semantic
                    values["category"] = result.category.value
                    for field, value in values.items():
                        setattr(match, field, value)
                    match.processing_status = ProcessingStatus.COMPLETED
                self._finish(row, BatchStatus.COMPLETED)

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
