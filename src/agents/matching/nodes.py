from uuid import UUID

from langgraph.runtime import Runtime

from src.agents.matching.state import MatchingContext, MatchingState
from src.exceptions.profile import ProfileNotFoundError
from src.infrastructure.models.preference_intent import PreferenceIntent
from src.infrastructure.models.vacancy import Vacancy
from src.schemas.vacancy_match import MatchingCandidate


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

    vacancies, preferences = await load_candidate_entities(state, runtime.context)

    return {
        "candidates": await runtime.context.matching_service.compare_candidates(
            profile,
            state.candidates,
            vacancies,
            preferences,
            limit=state.rerank_limit,
        ),
    }


async def save_matches(
    state: MatchingState,
    runtime: Runtime[MatchingContext],
) -> dict:
    results = [
        runtime.context.matching_service.build_result(
            profile_id=state.profile_id,
            vacancy_id=vacancy_id,
            preference_intent_id=candidate.preference_id,
            profile_comparison=candidate.profile_comparison,
            preference_comparison=candidate.preference_comparison,
            laya_comparison=candidate.laya_comparison,
            profile_rerank_score=candidate.profile_rerank_score,
            preference_rerank_score=candidate.preference_rerank_score,
            title_similarity=candidate.title_similarity,
            content_similarity=candidate.content_similarity,
            embedding_similarity=candidate.embedding_similarity,
        )
        for vacancy_id, candidate in state.candidates.items()
    ]
    results.sort(key=lambda result: result.total_score, reverse=True)
    saved_matches = await runtime.context.matching_service.save_results(results)
    return {"vacancy_match_ids": [vacancy_match.id for vacancy_match in saved_matches]}


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
