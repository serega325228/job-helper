from typing import TYPE_CHECKING

from src.schemas.vacancy import NormalizedVacancy, VacancySoftConditions

if TYPE_CHECKING:
    from src.infrastructure.models.preference_intent import PreferenceIntent
    from src.infrastructure.models.profile import Profile
    from src.infrastructure.models.vacancy import Vacancy


def _join(values: list[str]) -> str:
    return ", ".join(value.strip() for value in values if value.strip())


def _soft_conditions(
    vacancy: Vacancy | NormalizedVacancy,
) -> VacancySoftConditions:
    value = vacancy.soft_conditions
    if isinstance(value, VacancySoftConditions):
        return value
    return VacancySoftConditions.model_validate(value)


def build_vacancy_search_text(
    vacancy: Vacancy | NormalizedVacancy,
) -> str:
    """Build semantic vacancy content without SQL-filterable conditions."""
    soft = _soft_conditions(vacancy)
    sections: list[str] = []

    if soft.summary:
        sections.append(f"Role summary: {soft.summary.strip()}")
    if soft.required_skills:
        sections.append(f"Required skills: {_join(soft.required_skills)}")
    if soft.preferred_skills:
        sections.append(f"Preferred skills: {_join(soft.preferred_skills)}")
    if soft.requirements:
        sections.append(f"Requirements: {'; '.join(soft.requirements)}")
    if soft.responsibilities:
        sections.append(f"Responsibilities: {'; '.join(soft.responsibilities)}")
    if soft.industries:
        sections.append(f"Industries: {_join(soft.industries)}")
    if soft.benefits:
        sections.append(f"Benefits: {'; '.join(soft.benefits)}")
    if soft.additional_conditions:
        sections.append(
            "Additional context: " + "; ".join(soft.additional_conditions),
        )

    return "\n".join(sections) or f"Position: {vacancy.title.strip()}"


def build_preference_search_text(preference: PreferenceIntent) -> str:
    sections: list[str] = []

    if preference.description:
        sections.append(f"Target role context: {preference.description.strip()}")
    if preference.target_titles:
        sections.append(f"Target positions: {_join(preference.target_titles)}")
    if preference.required_skills:
        sections.append(f"Required skills: {_join(preference.required_skills)}")
    if preference.preferred_skills:
        sections.append(f"Preferred skills: {_join(preference.preferred_skills)}")
    if preference.preferred_industries:
        sections.append(
            f"Preferred industries: {_join(preference.preferred_industries)}",
        )
    if preference.preferred_companies:
        sections.append(
            f"Preferred companies: {_join(preference.preferred_companies)}",
        )

    return "\n".join(sections) or f"Target position: {preference.name.strip()}"


def build_preference_title_text(preference: PreferenceIntent) -> str:
    titles = _join(preference.target_titles)
    if titles:
        return f"{preference.name.strip()}: {titles}"
    return preference.name.strip()


def build_profile_search_text(profile: Profile) -> str:
    sections: list[str] = []

    if profile.profile_summary:
        sections.append(f"Candidate summary: {profile.profile_summary.strip()}")
    if profile.skills:
        sections.append(f"Candidate skills: {_join(profile.skills)}")
    if profile.experience:
        sections.append(f"Experience: {'; '.join(profile.experience)}")
    if profile.education:
        sections.append(f"Education: {'; '.join(profile.education)}")
    if profile.seniority:
        sections.append(f"Seniority: {profile.seniority}")

    return "\n".join(sections) or "Candidate profile has no structured data."
