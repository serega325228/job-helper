import asyncio
import hashlib
import re
from collections import defaultdict
from typing import ClassVar
from uuid import UUID

from src.infrastructure.laya.laya_provider import LayaProvider
from src.infrastructure.laya.questions import MATCH_QUESTIONS
from src.infrastructure.models.preference_intent import PreferenceIntent
from src.infrastructure.models.profile import Profile
from src.infrastructure.models.vacancy import Vacancy
from src.ports.embedder import Embedder
from src.ports.reranker import Reranker
from src.schemas.scoring import (
    LayaComparison,
    PreferenceComparison,
    ProfileComparison,
    VacancyRerankScores,
)
from src.schemas.vacancy import (
    EXPERIENCE_MINIMUMS,
    SENIORITY_RANKS,
    NormalizedVacancy,
    VacancyPreview,
    VacancySoftConditions,
)
from src.services.embedding_text import (
    build_preference_search_text,
    build_preference_title_text,
    build_profile_search_text,
    build_vacancy_search_text,
    soft_conditions,
)
from src.services.skill_canonicalization import SkillCanonicalizer
from src.services.vacancy_preview import VacancyPreviewEvaluator


class ScoringService:
    PROFILE_WEIGHTS: ClassVar[dict[str, float]] = {
        "required_skills": 0.55,
        "preferred_skills": 0.15,
        "experience": 0.20,
        "seniority": 0.10,
    }
    PREFERENCE_WEIGHTS: ClassVar[dict[str, float]] = {
        "title": 0.25,
        "required_skills": 0.20,
        "preferred_skills": 0.10,
        "industry": 0.10,
        "company": 0.08,
        "location": 0.10,
        "work_format": 0.07,
        "employment_type": 0.04,
        "seniority": 0.03,
        "salary": 0.03,
    }

    def __init__(
        self,
        reranker: Reranker,
        embedder: Embedder,
        skill_canonicalizer: SkillCanonicalizer,
        laya: LayaProvider,
        evaluator: VacancyPreviewEvaluator,
    ) -> None:
        self._reranker = reranker
        self._embedder = embedder
        self._skills = skill_canonicalizer
        self._laya = laya
        self._evaluator = evaluator

    def embedding_texts(self, entity: Vacancy | PreferenceIntent) -> list[str]:
        if isinstance(entity, PreferenceIntent):
            return [
                "Instruct: Given a job search query, retrieve relevant job vacancies"
                f"\nQuery: {text}"
                for text in (
                    build_preference_title_text(entity),
                    build_preference_search_text(entity),
                )
            ]
        return [entity.title, f"{entity.title}\n{build_vacancy_search_text(entity)}"]

    def embedding_fingerprint(self, entity: Vacancy | PreferenceIntent) -> str:
        return hashlib.sha256(
            repr((self._embedder.model_name, self.embedding_texts(entity))).encode()
        ).hexdigest()

    async def embed_entities(self, entities: list) -> dict:
        stale = [
            entity
            for entity in entities
            if entity.title_embedding is None
            or entity.content_embedding is None
            or entity.embedding_fingerprint != self.embedding_fingerprint(entity)
        ]
        if not stale:
            return {}
        vectors = await self._embedder.embed(
            [text for entity in stale for text in self.embedding_texts(entity)]
        )
        pairs = zip(vectors[::2], vectors[1::2], strict=True)
        return {
            entity.id: (title, content, self.embedding_fingerprint(entity))
            for entity, (title, content) in zip(stale, pairs, strict=True)
        }

    def compare_profile(
        self,
        profile: Profile,
        vacancy: Vacancy,
    ) -> ProfileComparison:
        soft = VacancySoftConditions.model_validate(vacancy.soft_conditions)
        profile_skills = self._skills.canonicalize_many(profile.skills)
        required = self._skills.canonicalize_many(soft.required_skills)
        preferred = self._skills.canonicalize_many(soft.preferred_skills)
        components: dict[str, float] = {}

        if required:
            components["required_skills"] = len(profile_skills & required) / len(
                required,
            )
        if preferred:
            components["preferred_skills"] = len(profile_skills & preferred) / len(
                preferred,
            )

        required_years = self._required_experience_years(vacancy.experience)
        if required_years is not None and profile.experience_years is not None:
            components["experience"] = self._ratio_score(
                profile.experience_years,
                required_years,
            )

        vacancy_seniority = self._seniority_rank(vacancy.seniority)
        profile_seniority = self._seniority_rank(profile.seniority)
        if vacancy_seniority is not None and profile_seniority is not None:
            gap = profile_seniority - vacancy_seniority
            components["seniority"] = 1.0 if gap >= 0 else max(0.0, 1.0 + gap * 0.5)

        matched = profile_skills & (required | preferred)
        missing = required - profile_skills
        return ProfileComparison(
            score=self._weighted_average(components, self.PROFILE_WEIGHTS),
            matched_skills=sorted(matched),
            missing_skills=sorted(missing),
            components=components,
        )

    def compare_preference(
        self,
        preference: PreferenceIntent,
        vacancy: Vacancy,
    ) -> PreferenceComparison:
        soft = VacancySoftConditions.model_validate(vacancy.soft_conditions)
        components: dict[str, float] = {}
        hard_constraints_passed = self._evaluator.matches_preference(
            VacancyPreview.model_validate(vacancy), preference
        )

        if preference.target_titles:
            components["title"] = max(
                self._lexical_title_score(target, vacancy.title)
                for target in preference.target_titles
            )

        vacancy_skills = self._skills.canonicalize_many(
            soft.required_skills + soft.preferred_skills,
        )
        required = self._skills.canonicalize_many(preference.required_skills)
        preferred = self._skills.canonicalize_many(preference.preferred_skills)
        if required:
            components["required_skills"] = len(vacancy_skills & required) / len(
                required,
            )
        if preferred:
            components["preferred_skills"] = len(vacancy_skills & preferred) / len(
                preferred,
            )

        self._add_overlap_component(
            components,
            "industry",
            preference.preferred_industries,
            soft.industries,
        )
        self._add_scalar_component(
            components,
            "company",
            preference.preferred_companies,
            vacancy.company_name,
        )
        self._add_overlap_component(
            components,
            "location",
            preference.locations,
            [value for value in (vacancy.city, vacancy.country) if value],
        )
        self._add_scalar_component(
            components,
            "work_format",
            preference.work_formats,
            vacancy.work_format,
        )
        self._add_scalar_component(
            components,
            "employment_type",
            preference.employment_types,
            vacancy.employment_type,
        )

        seniority_score = self._preference_seniority_score(preference, vacancy)
        if seniority_score is not None:
            components["seniority"] = seniority_score
        if (
            preference.salary_min is not None
            and vacancy.salary_to is not None
            and self._same_currency(preference, vacancy)
        ):
            components["salary"] = self._ratio_score(
                vacancy.salary_to,
                preference.salary_min,
            )

        score = self._weighted_average(components, self.PREFERENCE_WEIGHTS)
        if not hard_constraints_passed:
            score = 0.0
        return PreferenceComparison(
            score=score,
            hard_constraints_passed=hard_constraints_passed,
            components=components,
        )

    async def laya_match_vacancies(
        self,
        profile: Profile,
        preferences_by_vacancy: dict[UUID, PreferenceIntent],
        vacancies: dict[UUID, Vacancy],
        *,
        batch_size: int = 32,
    ) -> dict[UUID, LayaComparison]:
        if not vacancies:
            return {}

        missing_preferences = vacancies.keys() - preferences_by_vacancy.keys()
        if missing_preferences:
            raise ValueError(
                "Preferences are missing for vacancies: "
                f"{sorted(map(str, missing_preferences))}",
            )

        states: list[dict] = [
            self.make_laya_match_state(
                vacancy,
                preferences_by_vacancy[vacancy_id],
                profile,
            )
            for vacancy_id, vacancy in vacancies.items()
        ]

        results_raw = await self._laya.evaluate_batch(
            states,
            MATCH_QUESTIONS,
            batch_size=batch_size,
        )
        return {
            vacancy_id: LayaComparison(
                role_fit=result["answers"]["role_fit"]["choice"],
                skill_fit=result["answers"]["skill_fit"]["choice"],
            )
            for vacancy_id, result in zip(vacancies, results_raw, strict=True)
        }

    @staticmethod
    def make_laya_match_state(
        vacancy: Vacancy, preference: PreferenceIntent, profile: Profile
    ) -> dict:
        candidate_state = {
            "target_role": build_preference_title_text(preference),
            "skills": profile.skills,
            "experience": profile.experience,
            "summary": profile.profile_summary,
            "seniority": profile.seniority,
            "experience_years": profile.experience_years,
        }

        cond = soft_conditions(vacancy)

        vacancy_state = NormalizedVacancy.model_validate(
            vacancy, from_attributes=True
        ).model_dump(mode="json", exclude={"source", "external_id"})
        vacancy_state.update(
            {
                "requirements": cond.requirements,
                "required_skills": cond.required_skills,
                "preferred_skills": cond.preferred_skills,
            }
        )

        return {
            "candidate": candidate_state,
            "vacancy": vacancy_state,
        }

    async def rerank_vacancies(
        self,
        profile: Profile,
        vacancies: list[Vacancy],
        preferences_by_vacancy: dict[UUID, PreferenceIntent],
    ) -> dict[UUID, VacancyRerankScores]:
        if not vacancies:
            return {}

        missing_preference_ids = {
            vacancy.id
            for vacancy in vacancies
            if vacancy.id not in preferences_by_vacancy
        }
        if missing_preference_ids:
            raise ValueError(
                "Preferences are missing for vacancies: "
                f"{sorted(map(str, missing_preference_ids))}",
            )

        documents = [build_vacancy_search_text(vacancy) for vacancy in vacancies]
        profile_scores, preference_scores = await asyncio.gather(
            self._reranker.score(build_profile_search_text(profile), documents),
            self._rerank_preferences(
                vacancies,
                preferences_by_vacancy,
                documents,
            ),
        )
        return {
            vacancy.id: VacancyRerankScores(
                profile_score=profile_score,
                preference_score=preference_scores[vacancy.id],
            )
            for vacancy, profile_score in zip(
                vacancies,
                profile_scores,
                strict=True,
            )
        }

    async def _rerank_preferences(
        self,
        vacancies: list[Vacancy],
        preferences_by_vacancy: dict[UUID, PreferenceIntent],
        documents: list[str],
    ) -> dict[UUID, float]:
        groups: dict[UUID, list[int]] = defaultdict(list)
        preferences: dict[UUID, PreferenceIntent] = {}
        for index, vacancy in enumerate(vacancies):
            preference = preferences_by_vacancy[vacancy.id]
            groups[preference.id].append(index)
            preferences[preference.id] = preference

        scores: dict[UUID, float] = {}

        async def score_group(preference_id: UUID, indexes: list[int]) -> None:
            group_scores = await self._reranker.score(
                build_preference_search_text(preferences[preference_id]),
                [documents[index] for index in indexes],
            )
            for index, score in zip(indexes, group_scores, strict=True):
                scores[vacancies[index].id] = score

        await asyncio.gather(
            *(
                score_group(preference_id, indexes)
                for preference_id, indexes in groups.items()
            ),
        )
        return scores

    @staticmethod
    def _weighted_average(
        components: dict[str, float],
        weights: dict[str, float],
    ) -> float:
        available_weight = sum(weights[name] for name in components)
        if available_weight == 0:
            return 0.5
        return (
            sum(components[name] * weights[name] for name in components)
            / available_weight
        )

    @staticmethod
    def _ratio_score(actual: float, required: float) -> float:
        if required <= 0:
            return 1.0
        return min(1.0, max(0.0, actual / required))

    @staticmethod
    def _normalize_text(value: str | None) -> str:
        return " ".join((value or "").casefold().split())

    def _add_overlap_component(
        self,
        components: dict[str, float],
        name: str,
        expected: list[str],
        actual: list[str],
    ) -> None:
        expected_set = {self._normalize_text(value) for value in expected}
        actual_set = {self._normalize_text(value) for value in actual}
        if expected_set and actual_set:
            components[name] = len(expected_set & actual_set) / len(expected_set)

    def _add_scalar_component(
        self,
        components: dict[str, float],
        name: str,
        expected: list[str],
        actual: str | None,
    ) -> None:
        if expected and actual:
            expected_set = {self._normalize_text(value) for value in expected}
            components[name] = float(self._normalize_text(actual) in expected_set)

    @staticmethod
    def _required_experience_years(value: str | None) -> float | None:
        if not value:
            return None
        compact = re.sub(r"[^a-zA-Z0-9]", "", value).casefold()
        if compact in EXPERIENCE_MINIMUMS:
            return EXPERIENCE_MINIMUMS[compact]
        match = re.search(r"\d+(?:[.,]\d+)?", value)
        return float(match.group().replace(",", ".")) if match else None

    def _seniority_rank(self, value: str | None) -> int | None:
        normalized = self._normalize_text(value)
        return next(
            (rank for name, rank in SENIORITY_RANKS.items() if name in normalized),
            None,
        )

    def _preference_seniority_score(
        self,
        preference: PreferenceIntent,
        vacancy: Vacancy,
    ) -> float | None:
        actual = self._seniority_rank(vacancy.seniority)
        if actual is None:
            return None
        minimum = self._seniority_rank(preference.min_seniority)
        maximum = self._seniority_rank(preference.max_seniority)
        if minimum is None and maximum is None:
            return None
        return float(
            (minimum is None or actual >= minimum)
            and (maximum is None or actual <= maximum),
        )

    def _same_currency(
        self,
        preference: PreferenceIntent,
        vacancy: Vacancy,
    ) -> bool:
        return (
            preference.salary_currency is None
            or vacancy.salary_currency is None
            or self._normalize_text(preference.salary_currency)
            == self._normalize_text(vacancy.salary_currency)
        )

    def _lexical_title_score(self, expected: str, actual: str) -> float:
        expected_normalized = self._normalize_text(expected)
        actual_normalized = self._normalize_text(actual)
        if expected_normalized == actual_normalized:
            return 1.0
        if expected_normalized in actual_normalized:
            return 0.85
        expected_words = set(expected_normalized.split())
        actual_words = set(actual_normalized.split())
        if not expected_words:
            return 0.0
        return len(expected_words & actual_words) / len(expected_words)
