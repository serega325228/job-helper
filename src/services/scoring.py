import asyncio
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import ClassVar
from uuid import UUID

from src.infrastructure.models.preference_intent import PreferenceIntent
from src.infrastructure.models.profile import Profile
from src.infrastructure.models.vacancy import Vacancy
from src.infrastructure.reranker.vacancy_reranker import VacancyReranker
from src.schemas.vacancy import VacancySoftConditions
from src.schemas.vacancy_match import MatchCategory, VacancyMatchResult
from src.services.embedding import EmbeddingService
from src.services.embedding_text import (
    build_preference_search_text,
    build_preference_title_text,
    build_profile_search_text,
    build_vacancy_search_text,
)
from src.services.skill_canonicalization import SkillCanonicalizer

SENIORITY_RANKS = {
    "intern": 0,
    "internship": 0,
    "trainee": 0,
    "junior": 1,
    "middle": 2,
    "mid": 2,
    "senior": 3,
    "lead": 4,
    "principal": 5,
}

EXPERIENCE_MINIMUMS = {
    "noexperience": 0.0,
    "between1and3": 1.0,
    "between3and6": 3.0,
    "morethan6": 6.0,
}


@dataclass(frozen=True, slots=True)
class ProfileComparison:
    score: float
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    components: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PreferenceComparison:
    score: float
    hard_constraints_passed: bool
    components: dict[str, float] = field(default_factory=dict)


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
        reranker: VacancyReranker,
        embedding_service: EmbeddingService,
        skill_canonicalizer: SkillCanonicalizer,
        rerank_limit: int = 40,
    ) -> None:
        if rerank_limit < 1:
            raise ValueError("rerank_limit must be greater than zero")
        self._reranker = reranker
        self._embedding = embedding_service
        self._skills = skill_canonicalizer
        self._rerank_limit = rerank_limit

    async def update_preference_embeddings(
        self,
        preferences: list[PreferenceIntent],
    ) -> None:
        if not preferences:
            return

        texts = [
            text
            for preference in preferences
            for text in (
                build_preference_title_text(preference),
                build_preference_search_text(preference),
            )
        ]
        vectors = await asyncio.to_thread(self._embedding.embed_queries, texts)

        for index, preference in enumerate(preferences):
            preference.title_embedding = vectors[index * 2]
            preference.content_embedding = vectors[index * 2 + 1]

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
        vacancy_title = self._normalize_text(vacancy.title)
        company = self._normalize_text(vacancy.company_name)

        excluded_title = any(
            normalized in vacancy_title
            for value in preference.excluded_titles
            if (normalized := self._normalize_text(value))
        )
        excluded_company = company and any(
            normalized == company
            for value in preference.excluded_companies
            if (normalized := self._normalize_text(value))
        )
        salary_failed = (
            preference.salary_min is not None
            and vacancy.salary_to is not None
            and vacancy.salary_to < preference.salary_min
            and self._same_currency(preference, vacancy)
        )
        hard_constraints_passed = not (
            excluded_title or excluded_company or salary_failed
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

    def select_preference(
        self,
        vacancy: Vacancy,
        preferences: list[PreferenceIntent],
    ) -> PreferenceIntent:
        enabled = [preference for preference in preferences if preference.enabled]
        if not enabled:
            raise ValueError("At least one enabled preference intent is required")

        valid = [
            preference
            for preference in enabled
            if self.compare_preference(
                preference,
                vacancy,
            ).hard_constraints_passed
        ]
        selectable = valid or enabled

        def selection_score(preference: PreferenceIntent) -> float:
            structured = self.compare_preference(preference, vacancy).score
            title_semantic = self._cosine_similarity(
                preference.title_embedding,
                vacancy.title_embedding,
            )
            content_semantic = self._cosine_similarity(
                preference.content_embedding,
                vacancy.content_embedding,
            )
            semantic_components = {
                name: score
                for name, score in (
                    ("title", title_semantic),
                    ("content", content_semantic),
                )
                if score is not None
            }
            if semantic_components:
                semantic = self._weighted_average(
                    semantic_components,
                    {"title": 0.60, "content": 0.40},
                )
                return (semantic * 0.75 + structured * 0.25) * preference.weight
            return structured * preference.weight

        return max(selectable, key=selection_score)

    async def score(
        self,
        vacancies: list[Vacancy],
        profile: Profile,
        preferences: list[PreferenceIntent],
    ) -> list[VacancyMatchResult]:
        if not vacancies:
            return []

        profile_preferences = [
            preference
            for preference in preferences
            if preference.profile_id == profile.id
        ]

        candidates = []
        for vacancy in vacancies:
            preference = self.select_preference(vacancy, profile_preferences)
            profile_comparison = self.compare_profile(profile, vacancy)
            preference_comparison = self.compare_preference(preference, vacancy)
            if not preference_comparison.hard_constraints_passed:
                continue

            structured_score = self._geometric_score(
                profile_comparison.score,
                preference_comparison.score,
                first_weight=0.45,
            )
            candidates.append(
                (
                    vacancy,
                    preference,
                    profile_comparison,
                    preference_comparison,
                    structured_score,
                ),
            )

        candidates.sort(key=lambda item: item[4], reverse=True)
        candidates = candidates[:self._rerank_limit]
        if not candidates:
            return []

        documents = [build_vacancy_search_text(item[0]) for item in candidates]
        profile_scores = await self._reranker.score(
            build_profile_search_text(profile),
            documents,
        )
        preference_scores = await self._rerank_preferences(candidates, documents)

        results = [
            self._build_result(
                profile,
                candidate,
                profile_rerank_score,
                preference_rerank_score,
            )
            for candidate, profile_rerank_score, preference_rerank_score in zip(
                candidates,
                profile_scores,
                preference_scores,
                strict=True,
            )
        ]
        return sorted(results, key=lambda result: result.total_score, reverse=True)

    async def _rerank_preferences(
        self,
        candidates: list[tuple],
        documents: list[str],
    ) -> list[float]:
        groups: dict[UUID, list[int]] = defaultdict(list)
        preferences: dict[UUID, PreferenceIntent] = {}
        for index, candidate in enumerate(candidates):
            preference = candidate[1]
            groups[preference.id].append(index)
            preferences[preference.id] = preference

        scores = [0.0] * len(candidates)

        async def score_group(preference_id: UUID, indexes: list[int]) -> None:
            group_scores = await self._reranker.score(
                build_preference_search_text(preferences[preference_id]),
                [documents[index] for index in indexes],
            )
            for index, score in zip(indexes, group_scores, strict=True):
                scores[index] = score

        await asyncio.gather(
            *(
                score_group(preference_id, indexes)
                for preference_id, indexes in groups.items()
            ),
        )
        return scores

    def _build_result(
        self,
        profile: Profile,
        candidate: tuple,
        profile_rerank_score: float,
        preference_rerank_score: float,
    ) -> VacancyMatchResult:
        vacancy, preference, profile_comparison, preference_comparison, _ = candidate
        profile_fit = self._geometric_score(
            profile_comparison.score,
            profile_rerank_score,
            first_weight=0.55,
        )
        preference_fit = self._geometric_score(
            preference_comparison.score,
            preference_rerank_score,
            first_weight=0.55,
        )
        total_score = self._geometric_score(
            profile_fit,
            preference_fit,
            first_weight=0.45,
        )

        return VacancyMatchResult(
            profile_id=profile.id,
            vacancy_id=vacancy.id,
            preference_intent_id=preference.id,
            structured_profile_score=profile_comparison.score,
            structured_preference_score=preference_comparison.score,
            profile_rerank_score=profile_rerank_score,
            preference_rerank_score=preference_rerank_score,
            total_score=total_score,
            category=self._classify(total_score),
            component_scores={
                "profile": profile_comparison.components,
                "preference": preference_comparison.components,
                "intent_semantic": {
                    "title": self._cosine_similarity(
                        preference.title_embedding,
                        vacancy.title_embedding,
                    ),
                    "content": self._cosine_similarity(
                        preference.content_embedding,
                        vacancy.content_embedding,
                    ),
                },
            },
            matched_skills=profile_comparison.matched_skills,
            missing_skills=profile_comparison.missing_skills,
            reranker_model=self._reranker.model_name,
        )

    @staticmethod
    def _weighted_average(
        components: dict[str, float],
        weights: dict[str, float],
    ) -> float:
        available_weight = sum(weights[name] for name in components)
        if available_weight == 0:
            return 0.5
        return sum(
            components[name] * weights[name]
            for name in components
        ) / available_weight

    @staticmethod
    def _geometric_score(
        first: float,
        second: float,
        *,
        first_weight: float,
    ) -> float:
        if first <= 0 or second <= 0:
            return 0.0
        return first**first_weight * second ** (1 - first_weight)

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

    @staticmethod
    def _cosine_similarity(
        first: list[float] | None,
        second: list[float] | None,
    ) -> float | None:
        if first is None or second is None or len(first) != len(second):
            return None
        first_norm = math.sqrt(sum(value * value for value in first))
        second_norm = math.sqrt(sum(value * value for value in second))
        if first_norm == 0 or second_norm == 0:
            return None
        cosine = sum(a * b for a, b in zip(first, second, strict=True)) / (
            first_norm * second_norm
        )
        return min(1.0, max(0.0, cosine))

    @staticmethod
    def _classify(score: float) -> MatchCategory:
        if score >= 0.75:
            return MatchCategory.TARGET
        if score >= 0.55:
            return MatchCategory.STRETCH
        if score >= 0.35:
            return MatchCategory.FALLBACK
        return MatchCategory.REJECT
