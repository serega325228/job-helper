from uuid import UUID

from src.agents.resume.schemas import WizardTurnOutput
from src.exceptions.profile import AnalysisError, ProfileNotFoundError
from src.infrastructure.db.sqlalchemy_unit_of_work import SqlAlchemyUnitOfWork
from src.infrastructure.llm.llm import LLMProvider
from src.infrastructure.models.preference import Preference
from src.infrastructure.models.profile import AnalysisStatus, Profile
from src.prompts.resume_wizard import RESUME_WIZARD_ANALYZE_PROFILE_PROMPT
from src.repositories.prompt import PromptRepository
from src.schemas.profile import ProfileData
from src.schemas.resume import ResumeData


class ProfileService:
    def __init__(
        self,
        unit_of_work: SqlAlchemyUnitOfWork,
        prompt_repo: PromptRepository,
        llm: LLMProvider,
    ) -> None:
        self._uow = unit_of_work
        self._prompts = prompt_repo
        self._llm = llm

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
        *,
        enabled_only: bool = False,
    ) -> list[Preference]:
        async with self._uow as uow:
            return await uow.profiles.get_preferences_by_profile_id(
                profile_id, enabled_only=enabled_only
            )

    async def get_preferences_by_ids(
        self,
        profile_id: UUID,
        preference_ids: list[UUID],
    ) -> list[Preference]:
        async with self._uow as uow:
            return await uow.profiles.get_preferences_by_ids(profile_id, preference_ids)


    # Fix this too
    async def analyze_story(self, profile_id: UUID) -> Profile:
        async with self._uow as uow:
            profile = await uow.profiles.get_by_id(profile_id)
            if profile is None:
                raise ProfileNotFoundError(profile_id)

            raw_story = profile.raw_story
            profile.analysis_status = AnalysisStatus.PROCESSING

        try:
            result = await self._llm.complete(
                f"User story: {raw_story}",
            )
        except Exception:
            await self._set_analysis_status(profile_id, AnalysisStatus.FAILED)
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
            if not profile.preferences:
                profile.preferences.extend(
                    Preference(
                        profile_id=profile.id,
                        **intent.model_dump(mode="json"),
                    )
                    for intent in result.preferences
                )
            profile.analysis_status = AnalysisStatus.COMPLETED

        return profile

    async def analyze_profile_and_prepare_resume_question(
        self,
        profile_id: UUID,
        language: str,
        current_section: str,
        last_answer: str,
        resume_data: ResumeData,
        profile_data: ProfileData,
    ) -> WizardTurnOutput:
        try:
            profile = await self.get_profile(profile_id)
            if not profile:
                raise ProfileNotFoundError(profile_id)

            prompt = self._prompts.get("custom_key", RESUME_WIZARD_ANALYZE_PROFILE_PROMPT) # Get the fucking rid of prompt repo
            prompt = prompt[0].format(
                output_language=language,
                current_section=current_section,
                resume_json=resume_data.model_dump_json(),
                profile_json=profile_data.model_dump_json(),
                answer_text=last_answer,
            )

            return await self._llm.complete(prompt, schema=WizardTurnOutput)

        except Exception as e:
            raise AnalysisError(e)

    async def _set_analysis_status(
        self,
        profile_id: UUID,
        status: AnalysisStatus,
    ) -> None:
        async with self._uow as uow:
            profile = await uow.profiles.get_by_id(profile_id)
            if profile is not None:
                profile.analysis_status = status
