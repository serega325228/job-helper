from sqlalchemy import func, or_
from sqlalchemy.sql.elements import ColumnElement

from src.infrastructure.models.vacancy import Vacancy
from src.schemas.vacancy import VacancyHardFilters


def build_vacancy_hard_filter_conditions(
    filters: VacancyHardFilters,
) -> list[ColumnElement[bool]]:
    conditions: list[ColumnElement[bool]] = []
    if filters.sources:
        conditions.append(Vacancy.source.in_(filters.sources))
    if filters.statuses:
        conditions.append(Vacancy.status.in_(filters.statuses))
    if filters.area_ids:
        conditions.append(Vacancy.area_id.in_(filters.area_ids))
    if filters.countries:
        conditions.append(Vacancy.country.in_(filters.countries))
    if filters.cities:
        conditions.append(Vacancy.city.in_(filters.cities))
    if filters.company_names:
        conditions.append(Vacancy.company_name.in_(filters.company_names))
    if filters.excluded_company_names:
        conditions.append(
            or_(
                Vacancy.company_name.is_(None),
                Vacancy.company_name.not_in(filters.excluded_company_names),
            ),
        )
    if filters.work_formats:
        conditions.append(Vacancy.work_format.in_(filters.work_formats))
    if filters.employment_types:
        conditions.append(Vacancy.employment_type.in_(filters.employment_types))
    if filters.work_schedules:
        conditions.append(Vacancy.work_schedule.in_(filters.work_schedules))
    if filters.experience:
        conditions.append(Vacancy.experience.in_(filters.experience))
    if filters.seniorities:
        conditions.append(Vacancy.seniority.in_(filters.seniorities))
    if filters.salary_min is not None:
        conditions.append(
            func.coalesce(Vacancy.salary_to, Vacancy.salary_from) >= filters.salary_min,
        )
    if filters.salary_currency is not None:
        conditions.append(Vacancy.salary_currency == filters.salary_currency)
    if filters.salary_gross is not None:
        conditions.append(Vacancy.salary_gross == filters.salary_gross)
    if filters.published_after is not None:
        conditions.append(Vacancy.published_at >= filters.published_after)
    return conditions
