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

#i should mix this 2 steps in 1
async def embedding_search(
    state: MatchingState,
    runtime: Runtime[MatchingContext]
):
    result = await runtime.context.matching_service.search_by_preferences(
        state.profile_id,
        state.vacancy_ids_after_hard_filters,
        100, #idk where limits should be
        title_weight=0.4,
    )

    result = {res.vacancy_id: res for res in result}
    return {"vacancy_embedding_search_result": result}


async def compare_vacancies(
    state: MatchingState,
    runtime: Runtime[MatchingContext]
):
    profile = await runtime.context.profile_service.get_profile(state.profile_id)
    if profile is None:
        raise ProfileNotFoundError(state.profile_id)

    vacancies = await runtime.context.vacancy_service.get_vacancies_by_ids(state.vacancy_ids_after_hard_filters)
    if vacancies is None:
        ...
    #fix ts and make fine state, not that shit
    preferences = {preference.id: preference
        for preference in
            await runtime.context.profile_service.get_preferences_by_ids(
                profile.id,
                list([res.preference_id for res in state.vacancy_embedding_search_result.values()])
            )
    }

    comparisons: dict[UUID, tuple[ProfileComparison, PreferenceComparison]] = {}
    for vacancy in vacancies:
        profile_comparison = runtime.context.scoring_service.compare_profile(profile, vacancy)
        preference_comparison = runtime.context.scoring_service.compare_preference(
            preferences[state.vacancy_embedding_search_result[vacancy.id].preference_id],
            vacancy,
        )
        comparisons[vacancy.id] = (profile_comparison, preference_comparison)

    return {"compared_vacancies": comparisons}

async def rerank_vacancies(
    state: MatchingState,
    runtime: Runtime[MatchingContext]
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
