import unittest
from datetime import UTC, datetime
from unittest.mock import AsyncMock

from sqlalchemy.dialects import postgresql

from src.repositories.vacancy import VacancyRepository
from src.schemas.vacancy import VacancyHardFilters


class VacancyRepositoryTests(unittest.IsolatedAsyncioTestCase):
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
