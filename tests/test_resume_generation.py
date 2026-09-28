import json
import unittest
from unittest.mock import AsyncMock, Mock

from src.config.settings import RefinerSettings
from src.exceptions.config import ConfigError
from src.infrastructure.llm.llm import LLMProvider
from src.repositories.prompt import PromptRepository
from src.schemas.interview import InterviewPrepData
from src.schemas.llm import FeatureConfig
from src.schemas.resume import ImproveDiffResult, JobKeywords, ResumeChange, ResumeData
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


def features(unverified=False, **overrides) -> FeatureConfig:
    return FeatureConfig.model_validate(
        {
            "enable_cover_letter": True,
            "enable_outreach_message": True,
            "enable_interview_prep": True,
            "allow_unverified_skills": unverified,
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
            features=features(),
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
            original, invalid, "Python SQL Rust", keywords(), features=features()
        )
        self.assertEqual(result.resume, original)
        self.assertEqual(len(result.rejected_changes), len(invalid))
        self.assertEqual(result.applied_changes, [])
        self.assertFalse(any(item.status == "applied" for item in result.suggestions))

    def test_unverified_mode_requires_explicit_job_skills_and_labels_suggestions(self):
        original = resume()
        service = ImproverService(AsyncMock(spec=LLMProvider), SkillCanonicalizer())
        extracted = JobKeywords(required_skills=["Rust", "InventedSkill"])
        result = service.apply_diffs(
            original,
            [
                change("additional.technicalSkills", "add_skill", "Rust"),
                change("additional.technicalSkills", "add_skill", "InventedSkill"),
            ],
            "Rust developer",
            extracted,
            features=features(True),
        )
        self.assertEqual(result.unverified_skills, ["Rust"])
        self.assertEqual(len(result.applied_changes), 1)
        self.assertEqual(len(result.rejected_changes), 1)
        self.assertTrue(result.suggestions[0].unverified)
        self.assertIn("confirm", result.suggestions[0].suggestion)

    def test_unverified_mode_does_not_invent_demonstrated_experience(self):
        service = ImproverService(AsyncMock(spec=LLMProvider), SkillCanonicalizer())
        for mode in (False, True):
            original = resume()
            result = service.apply_diffs(
                original,
                [
                    change(
                        "summary",
                        "replace",
                        "Experienced Rust developer",
                        original.summary,
                    )
                ],
                "Rust developer",
                keywords(),
                features=features(mode),
            )
            self.assertEqual(result.resume, original)
            self.assertEqual(len(result.rejected_changes), 1)

    def test_strategies_and_term_boundaries(self):
        service = ImproverService(AsyncMock(spec=LLMProvider), SkillCanonicalizer())
        self.assertFalse(service.has_evidence("Go", resume()))
        self.assertFalse(service.keyword_in_text("Go", "ongoing development"))
        self.assertTrue(service.keyword_in_text("C++", "We use C++ and Python."))
        for strategy, edit in [
            ("nudge", change("additional.technicalSkills", "add_skill", "SQL")),
            (
                "keywords",
                change("workExperience[0].description", "append", "More reports."),
            ),
        ]:
            result = service.apply_diffs(
                resume(),
                [edit],
                "SQL",
                keywords(),
                features=features(),
                prompt_id=strategy,
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
            features=features(),
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
        refiner = RefinerService(llm, RefinerSettings(_env_file=None), improver)
        result = await refiner.refine_resume(
            original,
            master,
            "PostgreSQL developer",
            JobKeywords(required_skills=["PostgreSQL"]),
            features=features(),
        )
        self.assertIn("PostgreSQL", result.refined_data.additional.technicalSkills)
        self.assertEqual(result.keywords_injected, ["PostgreSQL"])
        self.assertEqual(result.to_stats().keywords_injected, 1)
        self.assertIs(llm.complete.await_args.kwargs["schema"], ImproveDiffResult)

    async def test_refiner_preserves_policy_and_restores_protected_data(self):
        master = resume()
        llm = AsyncMock(spec=LLMProvider)
        improver = ImproverService(llm, SkillCanonicalizer())
        settings = RefinerSettings(_env_file=None, enable_keyword_injection=False)
        for mode in (False, True):
            tailored = master.model_copy(deep=True)
            tailored.additional.technicalSkills.append("Rust")
            tailored.workExperience[0].company = "Invented company"
            result = await RefinerService(llm, settings, improver).refine_resume(
                tailored,
                master,
                "Python SQL Rust",
                keywords(),
                features=features(mode),
            )
            self.assertEqual(
                result.refined_data.workExperience[0].company, "Robust Inc"
            )
            self.assertEqual(
                "Rust" in result.refined_data.additional.technicalSkills, mode
            )
            self.assertEqual(result.unverified_skills, ["Rust"] if mode else [])
        llm.complete.assert_not_called()

    def test_phrase_cleanup_only_changes_prose(self):
        original = resume()
        original.summary = "Built robust tools."
        cleaned, removed = RefinerService.remove_ai_phrases(original)
        self.assertIn("robust", removed)
        self.assertEqual(cleaned.personalInfo, original.personalInfo)
        self.assertEqual(cleaned.workExperience[0].company, "Robust Inc")
        self.assertEqual(cleaned.workExperience[0].title, "Robust engineer")
        self.assertEqual(original.summary, "Built robust tools.")

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
