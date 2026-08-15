import unittest
from uuid import uuid4

from src.schemas.vacancy import VacancyHardFilters
from src.schemas.vacancy_match import MatchCategory, VacancyMatchResult
from src.services.vacancy_match import VacancyMatchService


class FakeVacancyMatchRepository:
    def __init__(self) -> None:
        self.item = None
        self.search_calls = []

    async def get_by_profile_and_vacancy(self, profile_id, vacancy_id):
        if (
            self.item is not None
            and self.item.profile_id == profile_id
            and self.item.vacancy_id == vacancy_id
        ):
            return self.item
        return None

    def add(self, vacancy_match) -> None:
        self.item = vacancy_match

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

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        return None

    async def flush(self) -> None:
        return None


class VacancyMatchServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_searches_with_hard_filters_in_same_repository_call(self) -> None:
        unit_of_work = FakeUnitOfWork()
        service = VacancyMatchService(unit_of_work)
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
        service = VacancyMatchService(unit_of_work)
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


if __name__ == "__main__":
    unittest.main()
