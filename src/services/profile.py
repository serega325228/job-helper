import json
from hashlib import sha256
from uuid import UUID

from src.agents.profile.schemas import ProfileOutput
from src.agents.resume.schemas import ResumeOutput
from src.agents.schemas import NextQuestion
from src.exceptions.profile import (
    AnalysisError,
    ProfileNotFoundError,
    ProfileWorkflowError,
)
from src.infrastructure.db.sqlalchemy_unit_of_work import SqlAlchemyUnitOfWork
from src.infrastructure.llm.llm import LLMProvider
from src.infrastructure.models.preference import Preference
from src.infrastructure.models.profile import AnalysisStatus, Profile
from src.prompts.profile import PROFILE_ANALYSIS_PROMPT
from src.prompts.resume_wizard import RESUME_WIZARD_ANALYZE_PROFILE_PROMPT
from src.repositories.prompt import PromptRepository
from src.schemas.preference import PreferenceData
from src.schemas.profile import ProfileData
from src.schemas.resume import ResumeData
from src.services.preference_catalog import (
    CATALOG_VERSION,
    ROLE_TITLES,
    preference_values,
    reconcile_preferences,
)


def profile_revision(profile: Profile) -> str:
    stored = dict(profile.processed_data or {})
    stored.pop("_wizard_save_key", None)
    return sha256(json.dumps(stored, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


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

    async def create(self, name: str, raw_story: str) -> Profile:
        profile = Profile(name=name, raw_data=raw_story)
        async with self._uow as uow:
            await uow.profiles.create(profile)
        return profile

    async def get_profile(self, profile_id: UUID) -> Profile | None:
        async with self._uow as uow:
            return await uow.profiles.get_by_id(profile_id)

    async def get_profile_data(self, profile_id: UUID) -> ProfileData:
        profile = await self.get_profile(profile_id)
        if profile is None:
            raise ProfileNotFoundError(profile_id)

        stored = dict(profile.processed_data or {})
        stored.pop("_wizard_save_key", None)
        # New saves retain the whole validated profile, including custom sections,
        # in the existing JSON column. Older rows may contain only extracted facts.
        if {"name", "raw_data", "processed_data", "profile_summary"} <= stored.keys():
            data = ProfileData.model_validate(stored)
            return data.model_copy(update={"raw_data": profile.raw_data})
        return ProfileData(
            name=profile.name or "",
            raw_data=profile.raw_data,
            processed_data=stored,
            profile_summary=profile.profile_summary or "",
            skills=profile.skills or [],
            experience=profile.experience or [],
            education=profile.education or [],
            seniority=profile.seniority,
            experience_years=profile.experience_years,
            contacts=profile.contacts or {},
        )

    async def get_preferences_by_profile_id(
        self, profile_id: UUID, *, enabled_only: bool = False
    ) -> list[Preference]:
        async with self._uow as uow:
            return await uow.profiles.get_preferences_by_profile_id(
                profile_id, enabled_only=enabled_only
            )

    async def get_preferences_by_ids(
        self, profile_id: UUID, preference_ids: list[UUID]
    ) -> list[Preference]:
        async with self._uow as uow:
            return await uow.profiles.get_preferences_by_ids(profile_id, preference_ids)

    async def analyze_data(
        self,
        profile_id: UUID,
        language: str,
        data: ProfileData,
        last_answer: str,
        *,
        last_question: NextQuestion | None = None,
        raw_answers: list[dict] | None = None,
        preference_text: str = "",
    ) -> ProfileOutput:
        try:
            async with self._uow as uow:
                profile = await uow.profiles.get_by_id(profile_id)
                if profile is None:
                    raise ProfileNotFoundError(profile_id)
                raw_data = profile.raw_data
                profile.analysis_status = AnalysisStatus.PROCESSING

            template, _ = self._prompts.get(
                "profile_analysis_prompt", PROFILE_ANALYSIS_PROMPT
            )
            out = await self._llm.complete(
                json.dumps(
                    {
                        "raw_data": raw_data,
                        "profile_data": data.model_dump(mode="json"),
                        "last_question": last_question.model_dump() if last_question else None,
                        "answer_text": last_answer,
                        "answer_history": raw_answers or [],
                        "preference_text": preference_text,
                        "preference_title_catalog": {
                            "version": CATALOG_VERSION,
                            "roles": [role.model_dump(mode="json") for role in ROLE_TITLES],
                        },
                    },
                    ensure_ascii=False,
                ),
                system_prompt=template.format(output_language=language),
                schema=ProfileOutput,
            )
            # Source preservation is an application invariant, not just a prompt rule.
            out.profile_data = out.profile_data.model_copy(update={"raw_data": raw_data})
            return out
        except Exception:
            await self._set_analysis_status(profile_id, AnalysisStatus.FAILED)
            raise

    async def update_profile(
        self,
        profile_id: UUID,
        data: ProfileData,
        preferences: list[PreferenceData] | None = None,
        *,
        expected_revision: str | None = None,
        save_key: str | None = None,
    ) -> Profile:
        async with self._uow as uow:
            profile = await uow.profiles.get_by_id(profile_id, lock=True)
            if profile is None:
                raise ProfileNotFoundError(profile_id)
            if save_key and (profile.processed_data or {}).get("_wizard_save_key") == save_key:
                return profile
            if expected_revision is not None and profile_revision(profile) != expected_revision:
                raise ProfileWorkflowError("wizard_profile_changed", 409)
            data = data.model_copy(update={"raw_data": profile.raw_data})
            profile.name = data.name
            profile.apply_analysis(
                profile_summary=data.profile_summary,
                skills=data.skills,
                experience=data.experience,
                education=data.education,
                seniority=data.seniority,
                experience_years=data.experience_years,
            )
            profile.contacts = data.contacts
            profile.processed_data = data.model_dump(mode="json")
            if save_key:
                profile.processed_data["_wizard_save_key"] = save_key
            # Only confirmed drafts reach this boundary. Reconcile again under the
            # profile row lock to handle retries and concurrent wizard threads.
            active = [PreferenceData.model_validate(item) for item in profile.preferences if item.enabled]
            additions = reconcile_preferences(
                preferences or [], active,
                occupied_titles=[item.general_title for item in profile.preferences],
            )
            profile.preferences.extend(
                Preference(profile_id=profile.id, **preference_values(item))
                for item in additions
            )
            profile.analysis_status = AnalysisStatus.COMPLETED
        return profile

    async def prepare_preference_drafts(
        self, profile_id: UUID, drafts: list[PreferenceData]
    ) -> list[PreferenceData]:
        existing = await self.get_preferences_by_profile_id(profile_id)
        return reconcile_preferences(
            drafts,
            [PreferenceData.model_validate(item) for item in existing if item.enabled],
            occupied_titles=[item.general_title for item in existing],
        )

    async def has_master_resume(self, profile_id: UUID) -> bool:
        async with self._uow as uow:
            return await uow.resumes.has_master_resume(profile_id)

    async def analyze_profile_and_prepare_resume_question(
        self,
        profile_id: UUID,
        language: str,
        last_question: NextQuestion | None,
        last_answer: str,
        resume_data: ResumeData,
        *,
        raw_answers: list[dict] | None = None,
    ) -> ResumeOutput:
        try:
            profile_data = await self.get_profile_data(profile_id)
            template, _ = self._prompts.get(
                "resume_wizard_profile_prompt", RESUME_WIZARD_ANALYZE_PROFILE_PROMPT
            )
            return await self._llm.complete(
                json.dumps(
                    {
                        "profile_data": profile_data.model_dump(mode="json"),
                        "resume_data": resume_data.model_dump(mode="json"),
                        "last_question": last_question.model_dump() if last_question else None,
                        "answer_text": last_answer,
                        "answer_history": raw_answers or [],
                    },
                    ensure_ascii=False,
                ),
                system_prompt=template.format(output_language=language),
                schema=ResumeOutput,
            )
        except ProfileNotFoundError:
            raise
        except Exception as error:
            raise AnalysisError(error) from error

    async def _set_analysis_status(
        self, profile_id: UUID, status: AnalysisStatus
    ) -> None:
        async with self._uow as uow:
            profile = await uow.profiles.get_by_id(profile_id)
            if profile is not None:
                profile.analysis_status = status
