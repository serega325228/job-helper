from uuid import UUID

from agents.matching.state import MatchingContext, MatchingState
from exceptions.profile import ProfileNotFoundError
from langgraph.runtime import Runtime
from schemas.scoring import PreferenceComparison, ProfileComparison


async def apply_hard_filters(
    state: MatchingState,
    runtime: Runtime[MatchingContext]
):
    vacancies = await runtime.context.vacancy_service.ids_by_hard_filters(state.hard_filters)

    return {"vacancy_ids_after_hard_filters": vacancies}

async def embedding_search(
    state: MatchingState,
    runtime: Runtime[MatchingContext]
):
    

async def compare_vacancy(
    state: MatchingState,
    runtime: Runtime[MatchingContext]
):
    profile = await runtime.context.profile_service.get_profile(state.profile_id)
    if profile is None:
        raise ProfileNotFoundError(state.profile_id)

    vacancies = await runtime.context.vacancy_service.get_vacancies_by_ids(state.vacancy_ids_after_hard_filters)
    if vacancies is None:
        ...

    preferences = await runtime.context.profile_service.get_preferences(profile.id)
    if preferences is None:
        ...

    comparisons: dict[UUID, tuple[ProfileComparison, PreferenceComparison]] = {}
    for vacancy in vacancies:
        profile_comparison = runtime.context.scoring_service.compare_profile(profile, vacancy)
        _, preference_comparison = runtime.context.scoring_service.select_preference(vacancy, preferences)
        comparisons[vacancy.id] = (profile_comparison, preference_comparison)

    return {"compared_vacancies": comparisons}

async def normalize_vacancies(
    state: CollectionState,
    runtime: Runtime[GraphContext]
):
    normalized_vacancies = []
    for offset in range(0, len(state.raw_vacancies), normalization_batch_size):
        batch = state.raw_vacancies[offset : offset + normalization_batch_size]
        normalized_batch = await runtime.context.vacancy_service.normalize_vacancies(batch)
        normalized_vacancies += normalized_batch

    return {"normalized_vacancies": normalized_vacancies}

async def save_vacancies(
    state: CollectionState,
    runtime: Runtime[GraphContext]
):
    saved_vacancies = await runtime.context.vacancy_service.save_vacancies(
        state.raw_vacancies, state.normalized_vacancies
    )

    return {"saved_vacancies": saved_vacancies}
