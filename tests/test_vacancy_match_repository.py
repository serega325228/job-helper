import unittest
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

from sqlalchemy.dialects import postgresql

from src.repositories.vacancy_match import VacancyMatchRepository
from src.schemas.vacancy import VacancyHardFilters


class VacancyMatchRepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_applies_hard_filters_during_preference_search(self) -> None:
        vacancy_id = uuid4()
        preference_id = uuid4()
        query_result = Mock()
        query_result.mappings.return_value = [
            {
                "vacancy_id": vacancy_id,
                "preference_id": preference_id,
                "title_similarity": 0.9,
                "content_similarity": 0.7,
                "combined_similarity": 0.78,
            },
        ]
        session = AsyncMock()
        session.execute.return_value = query_result
        repository = VacancyMatchRepository(session)
        profile_id = uuid4()

        result = await repository.search_by_preferences(
            profile_id,
            VacancyHardFilters(cities=["Москва"], work_formats=["remote"]),
            limit=50,
            title_weight=0.4,
            candidate_limit=100,
            per_preference_limit=40,
        )

        self.assertEqual(result[0].vacancy_id, vacancy_id)
        self.assertEqual(result[0].preference_id, preference_id)
        self.assertEqual(result[0].combined_similarity, 0.78)
        statement = session.execute.await_args.args[0]
        sql = str(statement.compile(dialect=postgresql.dialect()))
        self.assertEqual(sql.count("vacancies.city IN"), 2)
        self.assertEqual(sql.count("vacancies.work_format IN"), 2)
        self.assertEqual(sql.count("vacancies.status IN"), 2)
        self.assertIn("JOIN LATERAL", sql)
        self.assertIn("UNION", sql)
        self.assertIn("ORDER BY vacancies.title_embedding <=>", sql)
        self.assertIn("ORDER BY vacancies.content_embedding <=>", sql)
        self.assertNotIn("vacancies.id IN", sql)
        self.assertIn("row_number() OVER", sql)

    async def test_validates_lateral_search_limits(self) -> None:
        repository = VacancyMatchRepository(AsyncMock())
        profile_id = uuid4()
        filters = VacancyHardFilters()

        with self.assertRaises(ValueError):
            await repository.search_by_preferences(
                profile_id,
                filters,
                candidate_limit=0,
            )
        with self.assertRaises(ValueError):
            await repository.search_by_preferences(
                profile_id,
                filters,
                per_preference_limit=0,
            )


if __name__ == "__main__":
    unittest.main()
