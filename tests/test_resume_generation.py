import json
import unittest
from typing import cast
from unittest.mock import AsyncMock, Mock

from src.config.settings import RefinerSettings
from src.exceptions.config import ConfigError
from src.infrastructure.llm.llm import LLMProvider
from src.repositories.prompt import PromptRepository
from src.schemas.interview import InterviewPrepData
from src.schemas.llm import FeatureConfig
from src.schemas.resume import (
    ImproveDiffResult,
    JobKeywords,
    ProtectedClaimsReview,
    ResumeChange,
    ResumeData,
    Strategy,
)
from src.prompts.templates import IMPROVE_PROMPT_OPTIONS
from src.services.cover_letter import CoverLetterService
from src.services.improver import ImproverService
from src.services.interview_prep import InterviewPrepService
from src.services.refiner import RefinerService
from src.services.skill_canonicalization import SkillCanonicalizer


def resume() -> ResumeData:
    return ResumeData.model_validate(
        {
            "personalInfo": {"name": "Robust Person", "customContact": "keep"},
            "summary": "Python developer building useful tools.",
            "workExperience": [
                {
                    "id": 1,
                    "company": "Robust Inc",
                    "title": "Robust engineer",
                    "years": "Jan 2020 - May 2024",
                    "description": ["Used SQL to build reports."],
                    "descriptionStyles": ["plain"],
                    "customEntryMetadata": {"keep": True},
                }
            ],
            "personalProjects": [
                {
                    "id": "project",
                    "name": "Tool",
                    "description": ["Built a Python tool."],
                }
            ],
            "education": [
                {"id": 1, "degree": "BSc", "description": "Studied algorithms."}
            ],
            "additional": {
                "technicalSkills": ["Python", "Git"],
                "languages": ["English", "Russian", "English"],
                "certificationsTraining": ["Existing qualification"],
            },
            "customSections": {
                "notes": {"sectionType": "text", "text": "Volunteered tutoring."}
            },
            "sectionMeta": {"Go": {"visible": True}},
            "unknownMetadata": {"version": 4},
        }
    )


def features(**overrides) -> FeatureConfig:
    return FeatureConfig.model_validate(
        {
            "enable_cover_letter": True,
            "enable_outreach_message": True,
            "enable_interview_prep": True,
        }
        | overrides
    )


def keywords() -> JobKeywords:
    return JobKeywords(required_skills=["Python", "SQL", "Rust"])


def change(path, action, value, original=None) -> ResumeChange:
    return ResumeChange(
        path=path, action=action, value=value, original=original, reason="Relevant edit"
    )


class ResumeGenerationTest(unittest.IsolatedAsyncioTestCase):
    def test_typed_round_trip_preserves_unknown_fields(self):
        original = resume().model_dump(exclude_unset=True)
        self.assertEqual(
            ResumeData.model_validate(original).model_dump(exclude_unset=True), original
        )

    def test_valid_changes_preserve_identity_and_do_not_mutate_input(self):
        original = resume()
        before = original.model_dump()
        service = ImproverService(AsyncMock(spec=LLMProvider), SkillCanonicalizer())
        edits = [
            change(
                "summary",
                "replace",
                "Developer experienced in Python.",
                original.summary,
            ),
            change(
                "workExperience[0].description", "append", "Built maintainable reports."
            ),
            change(
                "education[0].description",
                "replace",
                "Studied computing algorithms.",
                "Studied algorithms.",
            ),
            change(
                "additional.languages", "reorder", ["Russian", "English", "English"]
            ),
            change("additional.technicalSkills", "add_skill", "SQL"),
        ]
        result = service.apply_diffs(
            original,
            edits,
            "Python SQL Rust",
            keywords(),
            prompt_id="full",
        )
        self.assertEqual(len(result.applied_changes), 5)
        self.assertEqual(result.rejected_changes, [])
        self.assertEqual(original.model_dump(), before)
        self.assertEqual(result.resume.personalInfo, original.personalInfo)
        self.assertEqual(result.resume.customSections, original.customSections)
        self.assertEqual(result.resume.workExperience[0].years, "Jan 2020 - May 2024")
        self.assertEqual(
            result.resume.workExperience[0].descriptionStyles, ["plain", "bullet"]
        )
        self.assertEqual(result.resume.model_extra, original.model_extra)
        self.assertEqual(
            result.resume.workExperience[0].model_extra,
            original.workExperience[0].model_extra,
        )
        self.assertTrue(
            any(
                item.status == "recommended" and "Rust" in item.suggestion
                for item in result.suggestions
            )
        )

    def test_rejects_unsafe_actions_paths_and_stale_originals(self):
        original = resume()
        service = ImproverService(AsyncMock(spec=LLMProvider), SkillCanonicalizer())
        invalid = [
            change("personalInfo.name", "replace", "Other", "Robust Person"),
            change("workExperience[0].company", "replace", "Other", "Robust Inc"),
            change("summary.garbage", "replace", "Other", original.summary),
            change("summary", "replace", "Other", None),
            change("summary", "replace", "Other", "stale text"),
            change("workExperience[99].description[0]", "replace", "Other", "missing"),
            change("education[0].description[0]", "replace", "Other", "S"),
            change("additional.technicalSkills", "append", "Rust"),
            change(
                "additional.certificationsTraining", "append", "Invented qualification"
            ),
            change("additional.technicalSkills", "reorder", ["Python", "Git", "Rust"]),
            change("additional.languages", "reorder", ["English", "Russian"]),
            change("additional.technicalSkills", "add_skill", "Rust"),
            change("additional.technicalSkills", "add_skill", "py"),
        ]
        result = service.apply_diffs(
            original, invalid, "Python SQL Rust", keywords()
        )
        self.assertEqual(result.resume, original)
        self.assertEqual(len(result.rejected_changes), len(invalid))
        self.assertEqual(result.applied_changes, [])
        self.assertFalse(any(item.status == "applied" for item in result.suggestions))

    def test_only_cheat_adds_unsupported_skills_and_labels_them(self):
        service = ImproverService(AsyncMock(spec=LLMProvider), SkillCanonicalizer())
        for mode in ("keywords", "full", "cheat"):
            result = service.apply_diffs(
                resume(),
                [
                    change("additional.technicalSkills", "add_skill", "Rust"),
                    change("additional.technicalSkills", "add_skill", "Kubernetes"),
                ],
                "Rust developer using Kubernetes",
                keywords(),
                prompt_id=mode,
            )
            self.assertEqual(len(result.applied_changes), 2 if mode == "cheat" else 0)
            self.assertEqual(result.unverified_skills, ["Rust", "Kubernetes"] if mode == "cheat" else [])
            if mode == "cheat":
                self.assertTrue(all(item.unverified for item in result.suggestions if item.status == "applied"))
                self.assertIn("confirm", result.suggestions[0].suggestion)

    def test_modes_control_invented_experience_metrics_and_bullets(self):
        service = ImproverService(AsyncMock(spec=LLMProvider), SkillCanonicalizer())
        for mode, expected in (("keywords", 0), ("full", 1), ("cheat", 3)):
            original = resume()
            result = service.apply_diffs(
                original,
                [
                    change("summary", "replace", "Experienced Rust developer", original.summary),
                    change("workExperience[0].description", "append", "Built maintainable reports."),
                    change("workExperience[0].description[0]", "replace", "Reduced reporting latency 40%.", original.workExperience[0].description[0]),
                ],
                "Rust SQL developer",
                keywords(),
                prompt_id=cast(Strategy, mode),
            )
            self.assertEqual(len(result.applied_changes), expected)
            self.assertEqual(result.resume.personalInfo, original.personalInfo)
            self.assertEqual(result.resume.education, original.education)
            self.assertEqual(result.resume.workExperience[0].company, "Robust Inc")
            if mode == "cheat":
                self.assertTrue(any("invented" in warning for warning in result.warnings))

    def test_strategies_and_term_boundaries(self):
        service = ImproverService(AsyncMock(spec=LLMProvider), SkillCanonicalizer())
        self.assertFalse(service.has_evidence("Go", resume()))
        self.assertFalse(service.keyword_in_text("Go", "ongoing development"))
        self.assertTrue(service.keyword_in_text("C++", "We use C++ and Python."))
        self.assertEqual([option["id"] for option in IMPROVE_PROMPT_OPTIONS], ["keywords", "full", "cheat"])
        result = service.apply_diffs(
            resume(),
            [change("workExperience[0].description", "append", "More reports.")],
            "SQL", keywords(), prompt_id="keywords",
        )
        self.assertEqual(len(result.rejected_changes), 1)

    async def test_improver_requests_typed_diffs_and_preserves_suggestions(self):
        original = resume()
        llm = AsyncMock(spec=LLMProvider)
        llm.complete.return_value = ImproveDiffResult(
            changes=[
                change("additional.technicalSkills", "add_skill", "SQL"),
            ],
            strategy_notes="Highlight demonstrated SQL experience.",
        )
        result = await ImproverService(llm, SkillCanonicalizer()).improve_resume(
            original,
            "Python SQL Rust",
            keywords(),
        )
        self.assertIs(llm.complete.await_args.kwargs["schema"], ImproveDiffResult)
        self.assertEqual(result.applied_changes[0].value, "SQL")
        self.assertTrue(result.suggestions)
        self.assertIn("SQL", result.strategy_notes)

    async def test_refiner_applies_typed_keyword_edits(self):
        original = resume()
        master = original.model_copy(deep=True)
        master.additional.technicalSkills.append("PostgreSQL")
        llm = AsyncMock(spec=LLMProvider)
        llm.complete.return_value = ImproveDiffResult(
            changes=[
                change("additional.technicalSkills", "add_skill", "PostgreSQL"),
            ]
        )
        improver = ImproverService(llm, SkillCanonicalizer())
        refiner = RefinerService(llm, RefinerSettings.model_construct(), improver)
        result = await refiner.refine_resume(
            original,
            master,
            "PostgreSQL developer",
            JobKeywords(required_skills=["PostgreSQL"]),
        )
        self.assertIn("PostgreSQL", result.refined_data.additional.technicalSkills)
        self.assertEqual(result.keywords_injected, ["PostgreSQL"])
        self.assertEqual(result.to_stats().keywords_injected, 1)
        self.assertIs(llm.complete.await_args.kwargs["schema"], ImproveDiffResult)

    async def test_refiner_preserves_policy_and_restores_protected_data(self):
        master = resume()
        llm = AsyncMock(spec=LLMProvider)
        improver = ImproverService(llm, SkillCanonicalizer())
        settings = RefinerSettings.model_construct(enable_keyword_injection=False)
        for mode in ("keywords", "full", "cheat"):
            tailored = master.model_copy(deep=True)
            tailored.additional.technicalSkills.append("Rust")
            tailored.workExperience[0].company = "Invented company"
            tailored.personalInfo.name = "Invented name"
            tailored.education[0].degree = "Invented degree"
            tailored.workExperience[0].years = "2030 - 2035"
            result = await RefinerService(llm, settings, improver).refine_resume(
                tailored,
                master,
                "Python SQL Rust",
                keywords(),
                prompt_id=mode,
            )
            self.assertEqual(result.refined_data.workExperience[0].company, "Robust Inc")
            self.assertEqual(result.refined_data.personalInfo, master.personalInfo)
            self.assertEqual(result.refined_data.education, master.education)
            self.assertEqual(result.refined_data.workExperience[0].years, master.workExperience[0].years)
            self.assertEqual("Rust" in result.refined_data.additional.technicalSkills, mode == "cheat")
            self.assertEqual(result.unverified_skills, ["Rust"] if mode == "cheat" else [])
        llm.complete.assert_not_called()

    def test_cheat_cannot_change_identity_metadata_or_create_entries(self):
        original = resume()
        service = ImproverService(AsyncMock(spec=LLMProvider), SkillCanonicalizer())
        edits = [
            change("personalInfo.name", "replace", "Other", original.personalInfo.name),
            change("workExperience[0].company", "replace", "Other", "Robust Inc"),
            change("workExperience[0].years", "replace", "2030", "Jan 2020 - May 2024"),
            change("education[0].degree", "replace", "PhD", "BSc"),
            change("education[0].description", "replace", "Invented academic award.", "Studied algorithms."),
            change("personalProjects[0].name", "replace", "Other", "Tool"),
            change("workExperience", "append", "Invented job"),
            change("personalProjects", "append", "Invented project"),
            change("additional.certificationsTraining", "append", "Invented certificate"),
        ]
        result = service.apply_diffs(original, edits, "Rust", keywords(), prompt_id="cheat")
        self.assertEqual(result.resume, original)
        self.assertEqual(len(result.rejected_changes), len(edits))

    def test_old_resume_keys_are_read_but_only_camel_case_is_serialized(self):
        original = resume().model_dump()
        legacy = resume().model_dump()
        for current, old in (
            ("personalInfo", "personal_info"),
            ("workExperience", "work_experience"),
            ("personalProjects", "personal_projects"),
            ("customSections", "custom_sections"),
            ("sectionMeta", "section_meta"),
        ):
            legacy[old] = legacy.pop(current)
        for section in ("work_experience", "personal_projects"):
            for entry in legacy[section]:
                entry["description_styles"] = entry.pop("descriptionStyles")
        for current, old in (
            ("technicalSkills", "technical_skills"),
            ("certificationsTraining", "certifications_training"),
        ):
            legacy["additional"][old] = legacy["additional"].pop(current)
        self.assertEqual(ResumeData.model_validate(legacy).model_dump(), original)

    async def test_cheat_reviews_protected_claims_inside_prose(self):
        original = resume()
        llm = AsyncMock(spec=LLMProvider)
        llm.complete.side_effect = [
            ImproveDiffResult(changes=[
                change("summary", "replace", "My name is Other. I hold a PhD from MIT.", original.summary),
                change("workExperience[0].description", "append", "Joined Google in 2025."),
                change("workExperience[0].description", "append", "Reduced latency 40% using Rust."),
                change("additional.technicalSkills", "add_skill", "Rust"),
            ]),
            ProtectedClaimsReview(rejected_indices=[0, 1]),
        ]
        result = await ImproverService(llm, SkillCanonicalizer()).improve_resume(
            original, "Rust developer", keywords(), prompt_id="cheat",
        )
        self.assertEqual(result.resume.summary, original.summary)
        self.assertEqual(len(result.rejected_changes), 2)
        self.assertEqual(result.resume.workExperience[0].description[-1], "Reduced latency 40% using Rust.")
        self.assertIn("Rust", result.resume.additional.technicalSkills)
        self.assertIs(llm.complete.await_args.kwargs["schema"], ProtectedClaimsReview)

    async def test_cheat_rejects_invalid_protected_review_indices(self):
        original = resume()
        llm = AsyncMock(spec=LLMProvider)
        llm.complete.return_value = ProtectedClaimsReview(rejected_indices=[99])
        service = ImproverService(llm, SkillCanonicalizer())
        proposal = ImproveDiffResult(changes=[
            change("additional.technicalSkills", "add_skill", "Rust"),
        ])
        with self.assertRaises(ValueError):
            await service.apply_proposal(original, proposal, "Rust", keywords(), prompt_id="cheat")
        self.assertNotIn("Rust", original.additional.technicalSkills)

    async def test_refinement_keeps_cheat_mode_and_does_not_clean_phrases(self):
        master = resume()
        tailored = master.model_copy(deep=True)
        tailored.summary = "Built robust tools."
        llm = AsyncMock(spec=LLMProvider)
        llm.complete.side_effect = [
            ImproveDiffResult(changes=[
                change("additional.technicalSkills", "add_skill", "Rust"),
                change("workExperience[0].description", "append", "Used Rust to reduce latency 40%."),
            ]),
            ProtectedClaimsReview(rejected_indices=[]),
        ]
        improver = ImproverService(llm, SkillCanonicalizer())
        result = await RefinerService(llm, RefinerSettings.model_construct(), improver).refine_resume(
            tailored, master, "Rust developer", JobKeywords(required_skills=["Rust"]), prompt_id="cheat",
        )
        self.assertIn("Rust", result.refined_data.additional.technicalSkills)
        self.assertIn("Used Rust to reduce latency 40%.", result.refined_data.workExperience[0].description)
        self.assertEqual(result.refined_data.summary, "Built robust tools.")
        assert result.alignment_report is not None
        self.assertTrue(result.alignment_report.is_aligned)
        self.assertEqual(result.unverified_skills, ["Rust"])
        self.assertNotIn("ai_phrases_removed", result.model_dump())
        self.assertNotIn("ai_phrases_removed", result.to_stats().model_dump())

    async def test_optional_generators_enforce_flags_and_use_typed_inputs(self):
        llm = AsyncMock(spec=LLMProvider)
        prompts = Mock(spec=PromptRepository)
        prompts.get.side_effect = lambda key, default: (default, False)
        cover_letter = CoverLetterService(llm, prompts)
        interview = InterviewPrepService(llm)
        disabled = features(
            enable_cover_letter=False,
            enable_outreach_message=False,
            enable_interview_prep=False,
        )
        for operation in (
            cover_letter.generate_cover_letter,
            cover_letter.generate_outreach_message,
            interview.generate_interview_prep,
        ):
            with self.assertRaises(ConfigError):
                await operation(resume(), "Python job", features=disabled)
        llm.complete.assert_not_called()
        llm.complete.return_value = "Hello employer"
        self.assertEqual(
            await cover_letter.generate_cover_letter(
                resume(), "Python job", features=features()
            ),
            "Hello employer",
        )
        self.assertNotIn("max_tokens", llm.complete.await_args.kwargs)
        llm.complete.return_value = InterviewPrepData(
            role_fit_analysis=[],
            resume_questions=[],
            project_follow_ups=[],
            skill_gaps=[],
            talking_points=[],
        )
        result = await interview.generate_interview_prep(
            resume(), "Python job", features=features()
        )
        self.assertIsInstance(result, InterviewPrepData)
        self.assertIs(llm.complete.await_args.kwargs["schema"], InterviewPrepData)

    def test_large_interview_input_remains_bounded_valid_json(self):
        original = resume()
        original.sectionMeta = {str(index): '\\"' * 1000 for index in range(500)}
        encoded = InterviewPrepService._serialize_resume_data_for_prompt(original)
        self.assertLessEqual(len(encoded), 30000)
        self.assertIsInstance(json.loads(encoded), dict)


if __name__ == "__main__":
    unittest.main()
