import unittest
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

from sqlalchemy.dialects import postgresql

from src.repositories.vacancy import VacancyRepository
from src.schemas.vacancy import VacancyHardFilters


class VacancyRepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_searches_title_and_content_embeddings(self) -> None:
        vacancy_id = uuid4()
        query_result = Mock()
        query_result.mappings.return_value = [
            {
                "vacancy_id": vacancy_id,
                "title_similarity": 0.9,
                "content_similarity": 0.7,
                "combined_similarity": 0.82,
            },
        ]
        session = AsyncMock()
        session.execute.return_value = query_result
        repository = VacancyRepository(session)

        result = await repository.search_by_embeddings(
            [1.0, 0.0],
            [0.0, 1.0],
            vacancy_ids=[vacancy_id],
            title_weight=0.6,
            limit=10,
            candidate_limit=40,
        )

        self.assertEqual(result[0].vacancy_id, vacancy_id)
        self.assertEqual(result[0].combined_similarity, 0.82)
        statement = session.execute.await_args.args[0]
        sql = str(statement.compile(dialect=postgresql.dialect()))
        self.assertIn("vacancies.title_embedding <=>", sql)
        self.assertIn("vacancies.content_embedding <=>", sql)
        self.assertIn("vacancies.id IN", sql)
        self.assertIn("UNION", sql)
        self.assertIn("embedding_candidates", sql)
        self.assertGreaterEqual(sql.count("ORDER BY"), 3)

    async def test_embedding_search_skips_query_for_empty_candidate_ids(self) -> None:
        session = AsyncMock()
        repository = VacancyRepository(session)

        result = await repository.search_by_embeddings(
            [1.0, 0.0],
            [0.0, 1.0],
            vacancy_ids=[],
        )

        self.assertEqual(result, [])
        session.execute.assert_not_awaited()

    async def test_embedding_search_validates_parameters(self) -> None:
        repository = VacancyRepository(AsyncMock())

        with self.assertRaises(ValueError):
            await repository.search_by_embeddings([], [1.0], limit=10)
        with self.assertRaises(ValueError):
            await repository.search_by_embeddings(
                [1.0],
                [1.0],
                title_weight=1.1,
            )
        with self.assertRaises(ValueError):
            await repository.search_by_embeddings(
                [1.0],
                [1.0],
                limit=10,
                candidate_limit=9,
            )

    async def test_builds_query_from_hard_filters(self) -> None:
        expected = object()
        session = AsyncMock()
        session.scalars.return_value = [expected]
        repository = VacancyRepository(session)
        filters = VacancyHardFilters(
            sources=["hh"],
            area_ids=["1"],
            cities=["Москва"],
            excluded_company_names=["Bad Employer"],
            work_formats=["remote"],
            employment_types=["full_time"],
            experience=["between1And3"],
            seniorities=["middle"],
            salary_min=200_000,
            salary_currency="RUR",
            salary_gross=True,
            published_after=datetime(2026, 8, 1, tzinfo=UTC),
        )

        result = await repository.list_by_hard_filters(filters, limit=40)

        self.assertEqual(result, [expected])
        statement = session.scalars.await_args.args[0]
        compiled = statement.compile(dialect=postgresql.dialect())
        sql = str(compiled)
        self.assertIn("vacancies.source IN", sql)
        self.assertIn("vacancies.status IN", sql)
        self.assertIn("vacancies.area_id IN", sql)
        self.assertIn("vacancies.city IN", sql)
        self.assertIn("vacancies.company_name NOT IN", sql)
        self.assertIn("vacancies.work_format IN", sql)
        self.assertIn("vacancies.employment_type IN", sql)
        self.assertIn("vacancies.experience IN", sql)
        self.assertIn("vacancies.seniority IN", sql)
        self.assertIn("coalesce(vacancies.salary_to, vacancies.salary_from)", sql)
        self.assertIn("vacancies.salary_currency =", sql)
        self.assertIn("vacancies.salary_gross = true", sql)
        self.assertIn("vacancies.published_at >=", sql)
        self.assertIn("ORDER BY vacancies.published_at DESC NULLS LAST", sql)
        self.assertEqual(compiled.params["param_1"], 40)

    async def test_rejects_invalid_limit(self) -> None:
        repository = VacancyRepository(AsyncMock())

        with self.assertRaises(ValueError):
            await repository.list_by_hard_filters(VacancyHardFilters(), limit=0)


if __name__ == "__main__":
    unittest.main()
