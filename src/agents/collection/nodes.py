from agents.collection.state import CollectionContext, CollectionState
from langgraph.runtime import Runtime


async def collect_raw_vacancies(
    state: CollectionState,
    runtime: Runtime[CollectionContext]
):
    raw_vacancies = []
    limit = 100
    fetch_concurrency = 8
    for source, query in zip(state.sources, state.queries):
        references = await runtime.context.vacancy_service.search_vacancies(
            source,
            query,
            limit=limit,
        )

        raw_vacancies += await runtime.context.vacancy_service.fetch_vacancies(
            source,
            references,
            concurrency=fetch_concurrency,
        )

    return {"raw_vacancies": raw_vacancies}

async def normalize_vacancies(
    state: CollectionState,
    runtime: Runtime[CollectionContext]
):
    normalized_vacancies = []
    normalization_batch_size = 5
    for offset in range(0, len(state.raw_vacancies), normalization_batch_size):
        batch = state.raw_vacancies[offset : offset + normalization_batch_size]
        normalized_batch = await runtime.context.vacancy_service.normalize_vacancies(batch)
        normalized_vacancies += normalized_batch

    return {"normalized_vacancies": normalized_vacancies}

async def save_vacancies(
    state: CollectionState,
    runtime: Runtime[CollectionContext]
):
    saved_vacancies = await runtime.context.vacancy_service.save_vacancies(
        state.raw_vacancies, state.normalized_vacancies
    )

    return {"saved_vacancies": saved_vacancies}
