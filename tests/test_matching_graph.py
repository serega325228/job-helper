import unittest
from uuid import uuid4

from src.agents.matching.graph import create_matching_graph
from src.agents.matching.state import MatchingContext
from src.infrastructure.models.preference_intent import PreferenceIntent
from src.infrastructure.models.profile import Profile
from src.infrastructure.models.vacancy import Vacancy
from src.schemas.scoring import (
    PreferenceComparison,
    ProfileComparison,
    VacancyRerankScores,
)
from src.schemas.vacancy import VacancyHardFilters
from src.schemas.vacancy_match import VacancyEmbeddingSearchResult


class FakeMatchingService:
    def __init__(self, result: VacancyEmbeddingSearchResult) -> None:
        self.result = result
        self.calls = []

    async def search_by_preferences(self, profile_id, hard_filters, **kwargs):
        self.calls.append((profile_id, hard_filters, kwargs))
        return [self.result]


class FakeProfileService:
    def __init__(self, profile: Profile, preference: PreferenceIntent) -> None:
        self.profile = profile
        self.preference = preference

    async def get_profile(self, profile_id):
        return self.profile if profile_id == self.profile.id else None

    async def get_preferences_by_ids(self, profile_id, preference_ids):
        if profile_id == self.profile.id and self.preference.id in preference_ids:
            return [self.preference]
        return []


class FakeVacancyService:
    def __init__(self, vacancy: Vacancy) -> None:
        self.vacancy = vacancy

    async def get_vacancies_by_ids(self, vacancy_ids):
        return [self.vacancy] if self.vacancy.id in vacancy_ids else []


class FakeScoringService:
    def __init__(self) -> None:
        self.rerank_calls = 0

    def compare_profile(self, profile, vacancy):
        return ProfileComparison(score=0.7)

    def compare_preference(self, preference, vacancy):
        return PreferenceComparison(score=0.8, hard_constraints_passed=True)

    async def rerank_vacancies(self, profile, vacancies, preferences_by_vacancy):
        self.rerank_calls += 1
        return {
            vacancy.id: VacancyRerankScores(
                profile_score=0.9,
                preference_score=0.85,
            )
            for vacancy in vacancies
        }


class MatchingGraphTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_compare_and_rerank_pipeline(self) -> None:
        profile_id = uuid4()
        vacancy_id = uuid4()
        preference_id = uuid4()
        profile = Profile(
            id=profile_id,
            name="Candidate",
            raw_story="Backend developer",
            skills=[],
            experience=[],
            education=[],
            contacts={},
        )
        preference = PreferenceIntent(
            id=preference_id,
            profile_id=profile_id,
            name="Backend",
        )
        vacancy = Vacancy(
            id=vacancy_id,
            source="hh",
            external_id="42",
            url="https://hh.ru/vacancy/42",
            title="Backend Developer",
            description="Backend systems",
            soft_conditions={},
            raw_payload={},
        )
        embedding_result = VacancyEmbeddingSearchResult(
            vacancy_id=vacancy_id,
            preference_id=preference_id,
            title_similarity=0.9,
            content_similarity=0.8,
            combined_similarity=0.84,
        )
        matching_service = FakeMatchingService(embedding_result)
        scoring_service = FakeScoringService()
        context = MatchingContext(
            profile_service=FakeProfileService(profile, preference),
            vacancy_service=FakeVacancyService(vacancy),
            scoring_service=scoring_service,
            matching_service=matching_service,
        )
        filters = VacancyHardFilters(cities=["Москва"])

        result = await create_matching_graph().ainvoke(
            {
                "profile_id": profile_id,
                "hard_filters": filters,
                "search_limit": 100,
                "rerank_limit": 40,
            },
            context=context,
        )

        candidate = result["candidates"][vacancy_id]
        self.assertEqual(candidate.preference_id, preference_id)
        self.assertIsNotNone(candidate.structured_score)
        self.assertEqual(candidate.profile_rerank_score, 0.9)
        self.assertEqual(candidate.preference_rerank_score, 0.85)
        self.assertEqual(scoring_service.rerank_calls, 1)
        self.assertEqual(matching_service.calls[0][1], filters)
        self.assertEqual(matching_service.calls[0][2]["candidate_limit"], 100)
        self.assertEqual(
            matching_service.calls[0][2]["per_preference_limit"],
            50,
        )


if __name__ == "__main__":
    unittest.main()
