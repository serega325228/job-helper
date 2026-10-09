import unittest
from datetime import UTC, datetime

from src.config.settings import Settings
from src.exceptions.vacancy import VacancyNormalizationError
from src.schemas.vacancy import NormalizedVacancy, RawVacancy
from src.services.vacancy import VacancyService


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


class VacancyServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_normalization_rejects_mismatched_external_ids(self) -> None:
        service = VacancyService(
            None,
            InvalidNormalizer(),
            Settings(_env_file=None),
        )
        raw = RawVacancy(
            source="hh",
            external_id="42",
            url="https://hh.ru/vacancy/42",
            title="Backend Engineer",
            fetched_at=datetime.now(UTC),
        )

        with self.assertRaises(VacancyNormalizationError):
            await service.normalize_vacancies([raw])


if __name__ == "__main__":
    unittest.main()
