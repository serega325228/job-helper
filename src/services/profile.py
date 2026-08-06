from uuid import UUID

from src.exceptions.profile import ProfileNotFoundError
from src.infrastructure.db.sqlalchemy_unit_of_work import SqlAlchemyUnitOfWork
from src.infrastructure.llm.profile_analyzer import ProfileAnalyzer
from src.infrastructure.models.profile import Profile


class ProfileService:
    def __init__(
        self,
        unit_of_work: SqlAlchemyUnitOfWork,
        analyzer: ProfileAnalyzer,
    ) -> None:
        self._uow = unit_of_work
        self._analyzer = analyzer

    async def create(
        self,
        name: str,
        raw_story: str,
    ) -> Profile:
        profile = Profile(name=name, raw_story=raw_story)

        async with self._uow as uow:
            await uow.profiles.create(profile)

        return profile

    async def analyze_story(self, profile_id: UUID) -> Profile:
        async with self._uow as uow:
            profile = await uow.profiles.get_by_id(profile_id)
            if profile is None:
                raise ProfileNotFoundError(profile_id)

            raw_story = profile.raw_story
            profile.analysis_status = "processing"

        try:
            result = await self._analyzer.analyze(raw_story)
        except Exception:
            await self._set_analysis_status(profile_id, "failed")
            raise

        async with self._uow as uow:
            profile = await uow.profiles.get_by_id(profile_id)
            if profile is None:
                raise ProfileNotFoundError(profile_id)

            profile.apply_analysis(
                profile_summary=result.summary,
                structured_data=result.model_dump(
                    exclude={"summary", "target_titles", "preferences"},
                    mode="json",
                ),
                preferences=result.preferences,
                target_titles=result.target_titles,
                contacts=profile.contacts,
            )
            profile.analysis_status = "completed"

        return profile

    async def _set_analysis_status(
        self,
        profile_id: UUID,
        status: str,
    ) -> None:
        async with self._uow as uow:
            profile = await uow.profiles.get_by_id(profile_id)
            if profile is not None:
                profile.analysis_status = status
