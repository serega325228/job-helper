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

        result = await repository.embedding_scores(
            profile_id,
            VacancyHardFilters(cities=["Москва"], work_formats=["remote"]),
            vacancy_ids=[vacancy_id],
            preference_ids=[preference_id],
            title_weight=0.4,
        )

        self.assertEqual(result[0].vacancy_id, vacancy_id)
        self.assertEqual(result[0].preference_id, preference_id)
        self.assertEqual(result[0].combined_similarity, 0.78)
        statement = session.execute.await_args.args[0]
        sql = str(statement.compile(dialect=postgresql.dialect()))
        self.assertEqual(sql.count("vacancies.city IN"), 1)
        self.assertEqual(sql.count("vacancies.work_format IN"), 1)
        self.assertEqual(sql.count("vacancies.status IN"), 1)
        self.assertNotIn("LATERAL", sql)
        self.assertNotIn("UNION", sql)
        self.assertIn("vacancies.id IN", sql)
        self.assertIn("preference_intents.id IN", sql)


if __name__ == "__main__":
    unittest.main()
