from uuid import UUID

from src.infrastructure.db.sqlalchemy_unit_of_work import SqlAlchemyUnitOfWork
from src.infrastructure.models.profile import Profile


class ProfileService:
    def __init__(
        self,
        unit_of_work: SqlAlchemyUnitOfWork,
    ):
        self._uow = unit_of_work

    async def create(
        self,
        name: str,
        raw_story: str,
    ):
        profile = Profile(
            name=name,
            raw_story=raw_story
        )

        async with self._uow as uow:
            await uow.profiles.create(profile)

        return profile

    async def analyze_story(
        self,
        profile_id: UUID,
    ):
        async with self._uow:
            profile = await self._uow.profiles.get_by_id(profile_id)

            if profile is None:
                raise ProfileNotFoundError(profile_id)

            raw_story = profile.raw_story
            profile.analysis_status = "processing"

        result = await self._analyzer.analyze(raw_story)

        async with self._uow:
            profile = await self._uow.profiles.get_by_id(profile_id)

            if profile is None:
                raise ProfileNotFroundError(profile_id)

            profile.profile_summary = result.summary
            profile.structured_data = result.structured_data
            profile.analysis_status = "completed"
