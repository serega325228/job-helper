import unittest
from datetime import UTC, datetime

from src.exceptions.vacancy import VacancyNormalizationError
from src.infrastructure.vacancy_sources.hh.source import HhVacancySource
from src.schemas.vacancy import (
    NormalizedVacancy,
    RawVacancy,
    VacancySearchQuery,
    VacancySoftConditions,
    WorkFormat,
)
from src.services.vacancy import VacancyService


class FakeHhClient:
    def __init__(self) -> None:
        self.search_params: list[dict] = []
        self.details_calls: list[str] = []

    async def search_vacancies(self, params: dict) -> dict:
        self.search_params.append(params)
        return {
            "pages": 1,
            "items": [
                {
                    "id": "42",
                    "name": "Backend Engineer",
                    "alternate_url": "https://hh.ru/vacancy/42",
                    "published_at": "2026-08-07T10:00:00+03:00",
                },
            ],
        }

    async def get_vacancy(self, vacancy_id: str) -> dict:
        self.details_calls.append(vacancy_id)
        return {
            "id": vacancy_id,
            "name": "Backend Engineer",
            "alternate_url": f"https://hh.ru/vacancy/{vacancy_id}",
            "description": "Go, PostgreSQL, remote",
            "published_at": "2026-08-07T10:00:00+03:00",
            "salary": {"from": 200_000, "to": 300_000, "currency": "RUR"},
        }


class FakeNormalizer:
    async def normalize(
        self,
        raw_vacancies: list[RawVacancy],
    ) -> list[NormalizedVacancy]:
        return [
            NormalizedVacancy(
                source=raw.source,
                external_id=raw.external_id,
                title=raw.title or "Unknown",
                company_name="Example",
                description="Backend development",
                city="Москва",
                work_format=WorkFormat.REMOTE,
                salary_from=200_000,
                salary_to=300_000,
                salary_currency="RUR",
                soft_conditions=VacancySoftConditions(
                    skills=["Go", "PostgreSQL"],
                    completeness_score=0.9,
                ),
            )
            for raw in raw_vacancies
        ]


class InvalidNormalizer:
    async def normalize(
        self,
        raw_vacancies: list[RawVacancy],
    ) -> list[NormalizedVacancy]:
        return [
            NormalizedVacancy(
                source="hh",
                external_id="unexpected",
                title="Wrong vacancy",
                description="Wrong mapping",
            ),
        ]


class FakeVacancyRepository:
    def __init__(self) -> None:
        self.items: dict[tuple[str, str], object] = {}

    async def get_by_external_keys(self, keys: set[tuple[str, str]]) -> list:
        return [self.items[key] for key in keys if key in self.items]

    def add_all(self, vacancies: list) -> None:
        for vacancy in vacancies:
            key = (vacancy.source, vacancy.external_id)
            self.items[key] = vacancy


class FakeUnitOfWork:
    def __init__(self) -> None:
        self.vacancies = FakeVacancyRepository()
        self.flush_count = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        return None

    async def flush(self) -> None:
        self.flush_count += 1


class VacancyServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_hh_pipeline_normalizes_and_upserts_vacancy(self) -> None:
        client = FakeHhClient()
        source = HhVacancySource(client)
        unit_of_work = FakeUnitOfWork()
        service = VacancyService(unit_of_work, FakeNormalizer())
        query = VacancySearchQuery(
            text="golang developer",
            area_ids=["1"],
            experience=["between1And3"],
            published_after=datetime(2026, 8, 1, tzinfo=UTC),
        )

        first = await service.ingest_vacancies(source, query)
        second = await service.ingest_vacancies(source, query)

        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        self.assertIs(first[0], second[0])
        self.assertEqual(len(unit_of_work.vacancies.items), 1)
        self.assertEqual(first[0].salary_from, 200_000)
        self.assertEqual(first[0].work_format, WorkFormat.REMOTE)
        self.assertEqual(first[0].soft_conditions["skills"], ["Go", "PostgreSQL"])
        self.assertEqual(first[0].raw_payload["id"], "42")
        self.assertEqual(client.details_calls, ["42", "42"])
        self.assertEqual(client.search_params[0]["area"], ["1"])
        self.assertEqual(
            client.search_params[0]["experience"],
            ["between1And3"],
        )
        self.assertIn("date_from", client.search_params[0])

    async def test_normalization_rejects_mismatched_external_ids(self) -> None:
        service = VacancyService(FakeUnitOfWork(), InvalidNormalizer())
        raw = RawVacancy(
            source="hh",
            external_id="42",
            url="https://hh.ru/vacancy/42",
            title="Backend Engineer",
            fetched_at=datetime.now(UTC),
        )

        with self.assertRaises(VacancyNormalizationError):
            await service.normalize_vacancies([raw])

    async def test_invalid_pipeline_limits_fail_fast(self) -> None:
        service = VacancyService(FakeUnitOfWork(), FakeNormalizer())
        source = HhVacancySource(FakeHhClient())
        query = VacancySearchQuery(text="developer")

        with self.assertRaises(ValueError):
            await service.ingest_vacancies(source, query, limit=0)

        with self.assertRaises(ValueError):
            await service.fetch_vacancies(source, [], concurrency=0)


if __name__ == "__main__":
    unittest.main()
