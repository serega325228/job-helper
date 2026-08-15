from uuid import UUID

from langgraph.runtime import Runtime

from src.agents.matching.schemas import MatchingCandidate
from src.agents.matching.state import MatchingContext, MatchingState
from src.exceptions.profile import ProfileNotFoundError
from src.infrastructure.models.preference_intent import PreferenceIntent
from src.infrastructure.models.vacancy import Vacancy


async def search_candidates(
    state: MatchingState,
    runtime: Runtime[MatchingContext],
) -> dict:
    results = await runtime.context.matching_service.search_by_preferences(
        state.profile_id,
        state.hard_filters,
        limit=state.search_limit,
        title_weight=state.title_weight,
        candidate_limit=state.ann_candidate_limit,
        per_preference_limit=state.per_preference_limit,
    )
    return {
        "candidates": {
            result.vacancy_id: MatchingCandidate(
                preference_id=result.preference_id,
                title_similarity=result.title_similarity,
                content_similarity=result.content_similarity,
                embedding_similarity=result.combined_similarity,
            )
            for result in results
        },
    }


async def compare_candidates(
    state: MatchingState,
    runtime: Runtime[MatchingContext],
) -> dict:
    if not state.candidates:
        return {"candidates": {}}

    profile = await runtime.context.profile_service.get_profile(state.profile_id)
    if profile is None:
        raise ProfileNotFoundError(state.profile_id)

    vacancies, preferences = await _load_candidate_entities(state, runtime)
    ranked: list[tuple[UUID, MatchingCandidate]] = []
    for vacancy_id, candidate in state.candidates.items():
        vacancy = vacancies[vacancy_id]
        preference = preferences[candidate.preference_id]
        profile_comparison = runtime.context.scoring_service.compare_profile(
            profile,
            vacancy,
        )
        preference_comparison = runtime.context.scoring_service.compare_preference(
            preference,
            vacancy,
        )
        if not preference_comparison.hard_constraints_passed:
            continue

        structured_score = _geometric_score(
            profile_comparison.score,
            preference_comparison.score,
            first_weight=0.45,
        )
        ranked.append(
            (
                vacancy_id,
                candidate.model_copy(
                    update={
                        "profile_comparison": profile_comparison,
                        "preference_comparison": preference_comparison,
                        "structured_score": structured_score,
                    },
                ),
            ),
        )

    ranked.sort(
        key=lambda item: item[1].structured_score or 0.0,
        reverse=True,
    )
    return {"candidates": dict(ranked[: state.rerank_limit])}


async def rerank_candidates(
    state: MatchingState,
    runtime: Runtime[MatchingContext],
) -> dict:
    if not state.candidates:
        return {"candidates": {}}

    profile = await runtime.context.profile_service.get_profile(state.profile_id)
    if profile is None:
        raise ProfileNotFoundError(state.profile_id)

    vacancies, preferences = await _load_candidate_entities(state, runtime)
    rerank_scores = await runtime.context.scoring_service.rerank_vacancies(
        profile,
        list(vacancies.values()),
        {
            vacancy_id: preferences[candidate.preference_id]
            for vacancy_id, candidate in state.candidates.items()
        },
    )
    return {
        "candidates": {
            vacancy_id: candidate.model_copy(
                update={
                    "profile_rerank_score": rerank_scores[vacancy_id].profile_score,
                    "preference_rerank_score": (
                        rerank_scores[vacancy_id].preference_score
                    ),
                },
            )
            for vacancy_id, candidate in state.candidates.items()
        },
    }

#remove ts out of here
async def _load_candidate_entities(
    state: MatchingState,
    runtime: Runtime[MatchingContext],
) -> tuple[dict[UUID, Vacancy], dict[UUID, PreferenceIntent]]:
    vacancies = {
        vacancy.id: vacancy
        for vacancy in await runtime.context.vacancy_service.get_vacancies_by_ids(
            list(state.candidates),
        )
    }
    preference_ids = list(
        {candidate.preference_id for candidate in state.candidates.values()},
    )
    preferences = {
        preference.id: preference
        for preference in await runtime.context.profile_service.get_preferences_by_ids(
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


def _geometric_score(
    first: float,
    second: float,
    *,
    first_weight: float,
) -> float:
    if first <= 0 or second <= 0:
        return 0.0
    return first**first_weight * second ** (1 - first_weight)
