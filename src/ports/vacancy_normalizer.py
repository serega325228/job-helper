from typing import Protocol

from src.schemas.vacancy import NormalizedVacancy, RawVacancy


class VacancyNormalizer(Protocol):
    async def normalize(
        self,
        raw_vacancies: list[RawVacancy],
    ) -> list[NormalizedVacancy]: ...
