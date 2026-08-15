from uuid import UUID

from src.agents.matching.state import MatchingContext, MatchingState
from src.infrastructure.models.preference_intent import PreferenceIntent
from src.infrastructure.models.vacancy import Vacancy


async def load_candidate_entities(
    state: MatchingState,
    context: MatchingContext,
) -> tuple[dict[UUID, Vacancy], dict[UUID, PreferenceIntent]]:
    vacancies = {
        vacancy.id: vacancy
        for vacancy in await context.vacancy_service.get_vacancies_by_ids(
            list(state.candidates),
        )
    }
    preference_ids = list(
        {candidate.preference_id for candidate in state.candidates.values()},
    )
    preferences = {
        preference.id: preference
        for preference in await context.profile_service.get_preferences_by_ids(
            state.profile_id,
            preference_ids,
        )
    }

    missing_vacancies = state.candidates.keys() - vacancies.keys()
    missing_preferences = set(preference_ids) - preferences.keys()
    if missing_vacancies or missing_preferences:
        raise RuntimeError(
            "Candidate entities are missing: "
            f"vacancies={sorted(map(str, missing_vacancies))}, "
            f"preferences={sorted(map(str, missing_preferences))}",
        )

    return vacancies, preferences
