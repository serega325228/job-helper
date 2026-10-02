import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

from src.schemas.scoring import (
    LayaComparison,
    PreferenceComparison,
    ProfileComparison,
    VacancyRerankScores,
)
from src.schemas.vacancy import VacancyHardFilters
from src.schemas.vacancy_match import (
    MatchCategory,
    MatchingCandidate,
    VacancyMatchResult,
)
from src.services.scoring import ScoringService
from src.services.vacancy_match import VacancyMatchService


class FakeVacancyMatchRepository:
    def __init__(self) -> None:
        self.items = []
        self.search_calls = []

    async def get_by_profile_and_vacancy(self, profile_id, vacancy_id):
        return next(
            (
                item
                for item in self.items
                if item.profile_id == profile_id and item.vacancy_id == vacancy_id
            ),
            None,
        )

    async def get_by_profile_and_vacancy_ids(self, profile_id, vacancy_ids):
        return [
            item
            for item in self.items
            if item.profile_id == profile_id and item.vacancy_id in vacancy_ids
        ]

    def add(self, vacancy_match) -> None:
        self.items.append(vacancy_match)

    def add_all(self, vacancy_matches) -> None:
        self.items.extend(vacancy_matches)

    async def search_by_preferences(
        self,
        profile_id,
        hard_filters,
        **kwargs,
    ):
        self.search_calls.append((profile_id, hard_filters, kwargs))
        return []


class FakeUnitOfWork:
    def __init__(self) -> None:
        self.vacancy_matches = FakeVacancyMatchRepository()
        self.flush_calls = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        return None

    async def flush(self) -> None:
        self.flush_calls += 1


class VacancyMatchServiceTests(unittest.IsolatedAsyncioTestCase):
    def test_builds_final_result_with_geometric_score(self) -> None:
        service = VacancyMatchService(FakeUnitOfWork(), None)
        result = service.build_result(
            profile_id=uuid4(),
            vacancy_id=uuid4(),
            preference_intent_id=uuid4(),
            profile_comparison=ProfileComparison(
                score=0.7,
                matched_skills=["python"],
                missing_skills=["redis"],
            ),
            preference_comparison=PreferenceComparison(
                score=0.8,
                hard_constraints_passed=True,
            ),
            laya_comparison=LayaComparison(role_fit="good", skill_fit="strong"),
            profile_rerank_score=0.9,
            preference_rerank_score=0.85,
            title_similarity=0.95,
            content_similarity=0.75,
            embedding_similarity=0.83,
        )

        expected_score = math.prod(
            (0.7**0.16, 0.8**0.16, 0.9**0.24, 0.85**0.24, (2 / 3)**0.1),
        )
        self.assertAlmostEqual(result.total_score, expected_score)
        self.assertEqual(result.category, MatchCategory.TARGET)
        self.assertEqual(result.matched_skills, ["python"])
        self.assertEqual(result.missing_skills, ["redis"])
        self.assertEqual(result.component_scores["semantic"]["combined"], 0.83)
        self.assertEqual(result.component_scores["laya"]["role_score"], 2 / 3)
        self.assertEqual(result.component_scores["laya"]["skill_score"], 1.0)

    async def test_filters_then_scores_with_laya_and_reranks_only_top_candidates(
        self,
    ) -> None:
        scoring = Mock(spec=ScoringService)
        service = VacancyMatchService(FakeUnitOfWork(), scoring)
        profile = SimpleNamespace(id=uuid4())
        vacancy_ids = [uuid4() for _ in range(3)]
        preferences = {
            preference_id: SimpleNamespace(id=preference_id)
            for preference_id in (uuid4(), uuid4())
        }
        preference_ids = list(preferences)
        candidates = {
            vacancy_id: MatchingCandidate(
                preference_id=preference_ids[index % 2],
                title_similarity=0.9,
                content_similarity=0.8,
                embedding_similarity=0.84,
            )
            for index, vacancy_id in enumerate(vacancy_ids)
        }
        vacancies = {
            vacancy_id: SimpleNamespace(id=vacancy_id)
            for vacancy_id in reversed(vacancy_ids)
        }
        scoring.compare_preference.side_effect = [
            PreferenceComparison(score=0.8, hard_constraints_passed=True),
            PreferenceComparison(score=0.0, hard_constraints_passed=False),
            PreferenceComparison(score=0.8, hard_constraints_passed=True),
        ]
        scoring.compare_profile.return_value = ProfileComparison(score=0.7)
        scoring.laya_match_vacancies.return_value = {
            vacancy_ids[0]: LayaComparison(role_fit="weak", skill_fit="good"),
            vacancy_ids[2]: LayaComparison(role_fit="strong", skill_fit="strong"),
        }
        scoring.rerank_vacancies.return_value = {
            vacancy_ids[2]: VacancyRerankScores(
                profile_score=0.9,
                preference_score=0.85,
            ),
        }

        result = await service.compare_candidates(
            profile,
            candidates,
            vacancies,
            preferences,
            limit=1,
        )

        self.assertEqual(list(result), [vacancy_ids[2]])
        self.assertEqual(result[vacancy_ids[2]].profile_rerank_score, 0.9)
        self.assertEqual(scoring.compare_profile.call_count, 2)
        scoring.laya_match_vacancies.assert_awaited_once_with(
            profile,
            {
                vacancy_id: preferences[preference_ids[0]]
                for vacancy_id in (vacancy_ids[0], vacancy_ids[2])
            },
            {
                vacancy_id: vacancies[vacancy_id]
                for vacancy_id in (vacancy_ids[0], vacancy_ids[2])
            },
        )
        scoring.rerank_vacancies.assert_awaited_once_with(
            profile,
            [vacancies[vacancy_ids[2]]],
            {vacancy_ids[2]: preferences[preference_ids[0]]},
        )
        self.assertTrue(
            all(
                candidate.profile_comparison is None
                for candidate in candidates.values()
            ),
        )

    async def test_all_rejected_candidates_skip_model_inference(self) -> None:
        scoring = Mock(spec=ScoringService)
        service = VacancyMatchService(FakeUnitOfWork(), scoring)
        vacancy_id, preference_id = uuid4(), uuid4()
        scoring.compare_preference.return_value = PreferenceComparison(
            score=0.0,
            hard_constraints_passed=False,
        )

        result = await service.compare_candidates(
            SimpleNamespace(id=uuid4()),
            {
                vacancy_id: MatchingCandidate(
                    preference_id=preference_id,
                    title_similarity=0.9,
                    content_similarity=0.8,
                    embedding_similarity=0.84,
                ),
            },
            {vacancy_id: SimpleNamespace(id=vacancy_id)},
            {preference_id: SimpleNamespace(id=preference_id)},
            limit=1,
        )

        self.assertEqual(result, {})
        scoring.compare_profile.assert_not_called()
        scoring.laya_match_vacancies.assert_not_awaited()
        scoring.rerank_vacancies.assert_not_awaited()

    async def test_searches_with_hard_filters_in_same_repository_call(self) -> None:
        unit_of_work = FakeUnitOfWork()
        service = VacancyMatchService(unit_of_work, None)
        profile_id = uuid4()
        filters = VacancyHardFilters(cities=["Москва"])

        result = await service.search_by_preferences(
            profile_id,
            filters,
            limit=50,
            title_weight=0.4,
        )

        self.assertEqual(result, [])
        self.assertEqual(
            unit_of_work.vacancy_matches.search_calls,
            [
                (
                    profile_id,
                    filters,
                    {
                        "limit": 50,
                        "title_weight": 0.4,
                        "candidate_limit": 100,
                        "per_preference_limit": 50,
                    },
                ),
            ],
        )

    async def test_save_result_upserts_profile_vacancy_match(self) -> None:
        unit_of_work = FakeUnitOfWork()
        service = VacancyMatchService(unit_of_work, None)
        profile_id = uuid4()
        vacancy_id = uuid4()
        first_intent_id = uuid4()
        second_intent_id = uuid4()
        first = VacancyMatchResult(
            profile_id=profile_id,
            vacancy_id=vacancy_id,
            preference_intent_id=first_intent_id,
            structured_profile_score=0.7,
            structured_preference_score=0.8,
            total_score=0.75,
            category=MatchCategory.STRETCH,
        )
        second = first.model_copy(
            update={
                "preference_intent_id": second_intent_id,
                "total_score": 0.9,
                "category": MatchCategory.TARGET,
            },
        )

        created = await service.save_result(first)
        updated = await service.save_result(second)

        self.assertIs(created, updated)
        self.assertEqual(updated.preference_intent_id, second_intent_id)
        self.assertEqual(updated.total_score, 0.9)
        self.assertEqual(updated.category, MatchCategory.TARGET.value)
        self.assertEqual(unit_of_work.flush_calls, 2)

    async def test_saves_multiple_results_in_one_unit_of_work(self) -> None:
        unit_of_work = FakeUnitOfWork()
        service = VacancyMatchService(unit_of_work, None)
        profile_id = uuid4()
        results = [
            VacancyMatchResult(
                profile_id=profile_id,
                vacancy_id=uuid4(),
                preference_intent_id=uuid4(),
                structured_profile_score=0.7,
                structured_preference_score=0.8,
                total_score=total_score,
                category=MatchCategory.STRETCH,
            )
            for total_score in (0.75, 0.70)
        ]

        saved = await service.save_results(results)

        self.assertEqual(len(saved), 2)
        self.assertEqual(len(unit_of_work.vacancy_matches.items), 2)
        self.assertEqual(unit_of_work.flush_calls, 1)


if __name__ == "__main__":
    unittest.main()
