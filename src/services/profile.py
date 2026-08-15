from uuid import UUID

from src.exceptions.profile import ProfileNotFoundError
from src.infrastructure.db.sqlalchemy_unit_of_work import SqlAlchemyUnitOfWork
from src.infrastructure.llm.profile_analyzer import ProfileAnalyzer
from src.infrastructure.models.preference_intent import PreferenceIntent
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

    async def get_profile(
        self,
        profile_id: UUID,
    ) -> Profile | None:
        async with self._uow as uow:
            profile = await uow.profiles.get_by_id(profile_id)

        return profile

    async def get_preferences_by_profile_id(
        self,
        profile_id: UUID,
    ) -> list[PreferenceIntent]:
        async with self._uow as uow:
            return await uow.profiles.get_preferences_by_profile_id(profile_id)

    async def get_preferences_by_ids(
        self,
        profile_id: UUID,
        preference_ids: list[UUID],
    ) -> list[PreferenceIntent]:
        async with self._uow as uow:
            return await uow.profiles.get_preferences_by_ids(profile_id, preference_ids)

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
                skills=result.skills,
                experience=result.experience,
                education=result.education,
                seniority=result.seniority,
                experience_years=result.experience_years,
            )
            if not profile.preference_intents:
                profile.preference_intents.extend(
                    PreferenceIntent(
                        profile_id=profile.id,
                        **intent.model_dump(mode="json"),
                    )
                    for intent in result.preference_intents
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
