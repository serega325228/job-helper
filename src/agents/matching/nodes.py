from agents.matching.state import MatchingContext, MatchingState
from exceptions.profile import ProfileNotFoundError
from langgraph.runtime import Runtime


async def compare_profile(
    state: MatchingState,
    runtime: Runtime[MatchingContext]
):
    profile = runtime.context.profile_service.get_profile(state.profile_id)
    if profile is None:
        raise ProfileNotFoundError(state.profile_id)

    runtime.context.scoring_service.compare_profile(profile)

    return {"raw_vacancies": raw_vacancies}

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
