from uuid import UUID

from langgraph.runtime import Runtime

from src.agents.matching.helpers import load_candidate_entities
from src.agents.matching.schemas import MatchingCandidate
from src.agents.matching.state import MatchingContext, MatchingState
from src.exceptions.profile import ProfileNotFoundError


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

        structured_score = runtime.context.matching_service.geometric_score(
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

    vacancies, preferences = await load_candidate_entities(state, runtime.context)
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
