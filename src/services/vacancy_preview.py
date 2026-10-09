import re
from datetime import UTC, datetime
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import UUID

from src.exceptions.profile import ProfileNotFoundError
from src.infrastructure.db.sqlalchemy_unit_of_work import SqlAlchemyUnitOfWork
from src.infrastructure.models.preference_intent import PreferenceIntent
from src.infrastructure.models.profile import Profile
from src.infrastructure.models.vacancy import VacancyBatch
from src.infrastructure.models.vacancy_preview import PreviewCollection
from src.schemas.vacancy import (
    EXPERIENCE_MINIMUMS,
    SENIORITY_RANKS,
    CollectionStatus,
    PreviewCollectionRequest,
    PreviewResponse,
    PreviewSelectionRequest,
    PreviewStatus,
    ProcessingStatus,
    VacancyHardFilters,
    VacancyPreview,
    VacancyScrapingQuery,
    VacancyStatus,
    WorkFormat,
)
from src.services.improver import ImproverService


def _normalize(value: str | None) -> str:
    return " ".join((value or "").casefold().split())


def _contains(value: str, text: str) -> bool:
    return ImproverService.keyword_in_text(_normalize(value), _normalize(text))


def _seniority_rank(value: str | None) -> int | None:
    ranks = {
        rank for name, rank in SENIORITY_RANKS.items() if _contains(name, value or "")
    }
    return ranks.pop() if len(ranks) == 1 else None


def _required_years(value: str | None) -> float | None:
    normalized = _normalize(value)
    native_minimum = EXPERIENCE_MINIMUMS.get(normalized)
    if native_minimum is not None:
        return native_minimum
    match = re.fullmatch(
        r"(?:at least\s+|minimum\s+|from\s+|от\s+|не менее\s+)?"
        r"(?P<minimum>\d+(?:[.,]\d+)?)\s*"
        r"(?:\+|[-–]\s*\d+(?:[.,]\d+)?)?\s*"
        r"(?:years?(?:\s+(?:of\s+)?experience)?|лет|года?)(?:\s+опыта)?",
        normalized,
    )
    return float(match["minimum"].replace(",", ".")) if match else None


class VacancyPreviewEvaluator:
    def accept(
        self,
        preview: VacancyPreview,
        profile: Profile,
        preferences: list[PreferenceIntent],
        filters: VacancyHardFilters,
        query: VacancyScrapingQuery,
    ) -> bool:
        if filters.sources and preview.source not in filters.sources:
            return False
        if filters.statuses and VacancyStatus.ACTIVE not in filters.statuses:
            return False

        for expected, actual in (
            (filters.company_names, preview.company_name),
            (filters.work_formats, preview.work_format),
            (filters.employment_types, preview.employment_type),
            (filters.experience, preview.experience),
            (filters.area_ids, preview.area_id),
            (filters.cities, preview.city),
            (filters.countries, preview.country),
            (filters.work_schedules, preview.work_schedule),
        ):
            if (
                expected
                and actual
                and _normalize(actual) not in map(_normalize, expected)
            ):
                return False
        if preview.company_name and _normalize(preview.company_name) in map(
            _normalize,
            filters.excluded_company_names,
        ):
            return False

        rank = _seniority_rank(preview.seniority or preview.title)
        allowed_ranks = [_seniority_rank(value) for value in filters.seniorities]
        if (
            rank is not None
            and allowed_ranks
            and None not in allowed_ranks
            and rank not in allowed_ranks
        ):
            return False
        required_years = _required_years(preview.experience)
        if (
            required_years is not None
            and profile.experience_years is not None
            and required_years > profile.experience_years
        ):
            return False
        if self._salary_fails(
            preview,
            filters.salary_min,
            filters.salary_currency,
            filters.salary_gross,
        ):
            return False

        published_at = preview.published_at
        for cutoff in (filters.published_after, query.published_after):
            if (
                published_at is not None
                and cutoff is not None
                and (published_at.tzinfo is None) == (cutoff.tzinfo is None)
                and published_at < cutoff
            ):
                return False

        text = "\n".join(
            value for value in (preview.title, preview.short_description) if value
        )
        if any(_contains(keyword, text) for keyword in query.excluded_keywords):
            return False
        if not preferences:
            return self.relevant(preview, [], query)
        return any(
            preference.enabled
            and self.matches_preference(preview, preference)
            and self.relevant(preview, [preference], query)
            for preference in preferences
        )

    @staticmethod
    def _salary_fails(
        preview: VacancyPreview,
        minimum: int | None,
        currency: str | None,
        gross: bool | None,
    ) -> bool:
        if (
            currency
            and preview.salary_currency
            and (_normalize(currency) != _normalize(preview.salary_currency))
        ):
            return True
        if (
            gross is not None
            and preview.salary_gross is not None
            and (gross != preview.salary_gross)
        ):
            return True
        return (
            minimum is not None
            and preview.salary_to is not None
            and preview.salary_to < minimum
        )

    def matches_preference(
        self,
        preview: VacancyPreview,
        preference: PreferenceIntent,
    ) -> bool:
        if any(_contains(title, preview.title) for title in preference.excluded_titles):
            return False
        if preview.company_name and _normalize(preview.company_name) in map(
            _normalize,
            preference.excluded_companies,
        ):
            return False
        for expected, actual in (
            (preference.work_formats, preview.work_format),
            (preference.employment_types, preview.employment_type),
        ):
            if (
                expected
                and actual
                and _normalize(actual) not in map(_normalize, expected)
            ):
                return False
        if (
            preference.locations
            and preview.city
            and preview.country
            and preview.work_format in {WorkFormat.ON_SITE, WorkFormat.HYBRID}
            and not any(
                _normalize(location)
                in {_normalize(preview.city), _normalize(preview.country)}
                for location in preference.locations
            )
        ):
            return False
        if self._salary_fails(
            preview, preference.salary_min, preference.salary_currency, None
        ):
            return False
        rank = _seniority_rank(preview.seniority or preview.title)
        minimum = _seniority_rank(preference.min_seniority)
        maximum = _seniority_rank(preference.max_seniority)
        return rank is None or (
            (minimum is None or rank >= minimum)
            and (maximum is None or rank <= maximum)
        )

    @staticmethod
    def relevant(
        preview: VacancyPreview,
        preferences: list[PreferenceIntent],
        query: VacancyScrapingQuery,
    ) -> bool:
        terms = [
            term for preference in preferences for term in preference.target_titles
        ] or [query.text]
        generic = set(SENIORITY_RANKS) | {
            "developer",
            "engineer",
            "разработчик",
            "инженер",
            "and",
            "or",
            "и",
            "the",
            "of",
        }
        # ponytail: title tokens miss semantic synonyms; extend the vocabulary if recall becomes insufficient.
        return any(
            _contains(term, preview.title)
            or any(
                _contains(word, preview.title)
                for word in (
                    [
                        word
                        for word in re.findall(r"\w[\w+#.-]*", _normalize(term))
                        if len(word) >= 2 and word not in generic
                    ]
                    or [
                        word
                        for word in re.findall(r"\w[\w+#.-]*", _normalize(term))
                        if len(word) >= 2 and word not in SENIORITY_RANKS
                    ]
                )
            )
            for term in terms
        )


class VacancyPreviewService:
    def __init__(
        self,
        unit_of_work: SqlAlchemyUnitOfWork,
        evaluator: VacancyPreviewEvaluator,
    ) -> None:
        self._uow = unit_of_work
        self._evaluator = evaluator

    async def prepare_collections(
        self, request: PreviewCollectionRequest, sources: dict
    ) -> list[UUID]:
        for names in (request.sources, request.preferences):
            if "all" in names and names != ["all"]:
                raise ValueError("Select either all or explicit names")
        async with self._uow as uow:
            if await uow.profiles.get_by_id(request.profile_id) is None:
                raise ProfileNotFoundError(request.profile_id)
            preferences = [
                preference
                for preference in await uow.profiles.get_preferences_by_profile_id(
                    request.profile_id, enabled_only=True
                )
                if (
                    request.preferences == ["all"]
                    or preference.name in request.preferences
                )
            ]
            if not preferences or (
                request.preferences != ["all"]
                and set(request.preferences)
                != {preference.name for preference in preferences}
            ):
                raise ValueError("Select at least one enabled, known preference")
            collections = []
            for source in await uow.previews.resolve_sources(request.sources):
                adapter_name = source.search_settings.get("source", source.name)
                adapter = sources.get(adapter_name)
                if adapter is None:
                    raise ValueError(
                        f"Source {source.name} requires a supported adapter in search_settings.source"
                    )
                url = request.vacancy_page_urls.get(source.name) or source.careers_url
                if url is None:
                    url = adapter.search_url(request.query, request.hard_filters)
                if (
                    adapter_name == "linkedin"
                    and source.name not in request.vacancy_page_urls
                ):
                    parts = urlsplit(str(url))
                    parameters = dict(parse_qsl(parts.query))
                    parameters.update(
                        parse_qsl(
                            urlsplit(
                                adapter.search_url(request.query, request.hard_filters)
                            ).query
                        )
                    )
                    url = urlunsplit(parts._replace(query=urlencode(parameters)))
                if adapter_name == "linkedin":
                    adapter._validate_page_url(str(url))
                else:
                    adapter.validate_search_url(str(url))
                collections.append(
                    PreviewCollection(
                        profile_id=request.profile_id,
                        source_id=source.id,
                        source=adapter_name,
                        vacancy_page_url=str(url),
                        preference_ids=[
                            str(preference.id) for preference in preferences
                        ],
                        query=request.query.model_dump(mode="json"),
                        hard_filters=request.hard_filters.model_dump(mode="json"),
                    )
                )
            uow.previews.add_collections(collections)
            await uow.flush()
        return [collection.id for collection in collections]

    async def list_previews(self, profile_id: UUID, limit: int) -> list:
        async with self._uow as uow:
            return [
                PreviewResponse(
                    **VacancyPreview.model_validate(preview).model_dump(),
                    id=collected.id,
                    status=collected.status,
                )
                for collected, preview in await uow.previews.get_collected(
                    profile_id=profile_id, limit=limit
                )
            ]

    async def save_scraped(
        self, collection_id: UUID, previews: list[VacancyPreview], token: UUID
    ) -> list[UUID]:
        async with self._uow as uow:
            collection = await uow.previews.get_collection(collection_id, lock=True)
            if (
                collection is None
                or collection.processing_token != token
                or collection.status != CollectionStatus.PENDING
            ):
                return []
            if any(preview.source != collection.source for preview in previews):
                raise ValueError("Preview source does not match its collection")
            collection.updated_at = datetime.now(UTC)
            saved = await uow.previews.save_new(collection_id, previews)
        return saved

    async def filter_previews(self, preview_ids: list[UUID]) -> None:
        async with self._uow as uow:
            groups = {}
            for collected, preview in await uow.previews.get_collected(
                ids=preview_ids, status=PreviewStatus.PENDING_FILTER, lock=True
            ):
                groups.setdefault(collected.collection_id, []).append(
                    (collected, preview)
                )
            for collection_id, group in groups.items():
                collection = await uow.previews.get_collection(collection_id)
                profile = await uow.profiles.get_by_id(collection.profile_id)
                try:
                    preference_ids = [
                        UUID(value) for value in collection.preference_ids
                    ]
                    query = VacancyScrapingQuery.model_validate(collection.query)
                    filters = VacancyHardFilters.model_validate(collection.hard_filters)
                except ValueError as error:
                    for collected, _ in group:
                        collected.status = PreviewStatus.FAILED
                        collected.error = str(error)
                    continue
                preferences = await uow.profiles.get_preferences_by_ids(
                    collection.profile_id,
                    preference_ids,
                    enabled_only=True,
                )
                for collected, preview in group:
                    try:
                        if profile is None:
                            raise ProfileNotFoundError(collection.profile_id)
                        accepted = preferences and self._evaluator.accept(
                            VacancyPreview.model_validate(preview),
                            profile,
                            preferences,
                            filters,
                            query,
                        )
                    except (
                        ValueError,
                        TypeError,
                        AttributeError,
                        ProfileNotFoundError,
                    ) as error:
                        collected.status = PreviewStatus.FAILED
                        collected.error = str(error)
                    else:
                        collected.status = (
                            PreviewStatus.READY if accepted else PreviewStatus.REJECTED
                        )
                        collected.error = None

    async def select_for_details(self, request: PreviewSelectionRequest) -> list[UUID]:
        async with self._uow as uow:
            profile = await uow.profiles.get_by_id(request.profile_id, lock=True)
            if profile is None:
                raise ProfileNotFoundError(request.profile_id)
            selected, preference_ids, collections = [], set(), {}
            seen = set()
            for collected, preview in await uow.previews.get_collected(
                profile_id=request.profile_id,
                ids=request.preview_ids,
                status=PreviewStatus.READY,
                lock=True,
            ):
                if preview.id in seen:
                    continue
                if collected.collection_id not in collections:
                    collection = await uow.previews.get_collection(
                        collected.collection_id
                    )
                    preferences = await uow.profiles.get_preferences_by_ids(
                        request.profile_id,
                        [UUID(value) for value in collection.preference_ids],
                        enabled_only=True,
                    )
                    collections[collected.collection_id] = (collection, preferences)
                collection, preferences = collections[collected.collection_id]
                if not preferences or not self._evaluator.accept(
                    VacancyPreview.model_validate(preview),
                    profile,
                    preferences,
                    request.hard_filters,
                    VacancyScrapingQuery.model_validate(collection.query),
                ):
                    continue
                selected.append((collected, preview))
                seen.add(preview.id)
                preference_ids.update(str(preference.id) for preference in preferences)
                if len(selected) >= request.limit:
                    break
            if not selected:
                return []
            vacancies = await uow.vacancies.create_from_previews(
                [
                    {
                        "preview_id": preview.id,
                        "source": preview.source,
                        "external_id": preview.external_id,
                        "url": preview.url,
                        "title": preview.title,
                        "company_name": preview.company_name,
                        "normalized_company": _normalize(preview.company_name) or None,
                        "normalized_title": _normalize(preview.title),
                        "description": "",
                        "processing_status": ProcessingStatus.PENDING_SCRAPE,
                    }
                    for _, preview in selected
                ]
            )
            busy = await uow.vacancies.busy_vacancy_ids(request.profile_id)
            vacancies = [
                vacancy
                for vacancy in vacancies
                if vacancy.id not in busy
                and vacancy.processing_status
                not in {ProcessingStatus.FAILED, ProcessingStatus.REMOVED}
            ]
            if not vacancies:
                return []
            by_key = {
                (vacancy.source, vacancy.external_id): vacancy for vacancy in vacancies
            }
            context = {}
            for collected, preview in selected:
                key = (preview.source, preview.external_id)
                if key not in by_key:
                    continue
                collection, preferences = collections[collected.collection_id]
                scope = context.setdefault(
                    str(collection.id),
                    {
                        "vacancy_ids": [],
                        "preference_ids": [
                            str(preference.id) for preference in preferences
                        ],
                        "hard_filters": collection.hard_filters,
                    },
                )
                scope["vacancy_ids"].append(str(by_key[key].id))
            batch = VacancyBatch(
                selection_context=context,
                profile_id=request.profile_id,
                vacancy_ids=[str(vacancy.id) for vacancy in vacancies],
                preference_ids=sorted(preference_ids),
                hard_filters=request.hard_filters.model_dump(mode="json"),
                search_limit=request.search_limit,
                rerank_limit=request.rerank_limit,
                title_weight=request.title_weight,
            )
            uow.vacancies.add_batch(batch)
            keys = {(vacancy.source, vacancy.external_id) for vacancy in vacancies}
            for collected, preview in selected:
                if (preview.source, preview.external_id) in keys:
                    collected.status = PreviewStatus.SELECTED
            await uow.flush()
        return [vacancy.id for vacancy in vacancies]
