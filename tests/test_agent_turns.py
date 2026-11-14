import unittest
from types import SimpleNamespace
from typing import Any, Literal, cast
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import ValidationError

from src.agents.profile.graph import create_profile_agent
from src.agents.profile.nodes import ask_user as ask_profile_user
from src.agents.profile.schemas import ProfileOutput
from src.agents.profile.state import ProfileContext, ProfileState
from src.agents.resume.graph import create_resume_agent
from src.agents.resume.nodes import ask_user as ask_resume_user
from src.agents.resume.schemas import ResumeOutput
from src.agents.resume.state import MasterResumeContext, MasterResumeState
from src.agents.schemas import FollowUpOutput, NextQuestion
from src.schemas.profile import ProfileData
from src.schemas.resume import PersonalInfo, ResumeData


def profile_data(name="Ada"):
    return ProfileData(name=name, raw_data="Original story", processed_data={}, profile_summary="")


def resume_data(name=None):
    return ResumeData(personalInfo=PersonalInfo(name=name))


class FollowUpContractTests(unittest.TestCase):
    def test_missing_fields_must_be_explicit(self):
        with self.assertRaises(ValidationError):
            FollowUpOutput.model_validate({"next_question": None})

    def test_empty_list_requires_no_question(self):
        result = FollowUpOutput(missing_fields=[], next_question=None)
        self.assertEqual(result.missing_fields, [])
        with self.assertRaises(ValidationError):
            FollowUpOutput.model_validate({
                "missing_fields": [], "next_question": {"field": "name", "text": "Name?"}
            })

    def test_nonempty_list_requires_matching_question(self):
        for question in (
            None,
            {"field": "skills", "text": "Skills?"},
            {"field": "name", "text": " "},
        ):
            with self.subTest(question=question), self.assertRaises(ValidationError):
                FollowUpOutput.model_validate({"missing_fields": ["name"], "next_question": question})
        result = FollowUpOutput(
            missing_fields=["name"], next_question=NextQuestion(field="name", text="Name?")
        )
        assert result.next_question is not None
        self.assertEqual(result.next_question.field, "name")

    def test_legacy_boolean_and_empty_paths_are_rejected(self):
        for payload in (
            {"missing_fields": [], "next_question": None, "is_complete": True},
            {"missing_fields": [" "], "next_question": None},
        ):
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                FollowUpOutput.model_validate(payload)

    def test_initial_states_need_no_fake_question_or_completion_flag(self):
        profile = ProfileState(profile_id=uuid4(), language="en")
        resume = MasterResumeState(profile_id=uuid4(), language="en")
        self.assertIsNone(profile.next_question)
        self.assertIsNone(resume.next_question)
        self.assertIsNone(resume.resume.personalInfo.name)
        self.assertNotIn("is_complete", ProfileState.model_fields)
        self.assertNotIn("is_complete", MasterResumeState.model_fields)


class AgentLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_answer_nodes_keep_gaps_and_record_actual_question(self):
        question = NextQuestion(field="name", text="What is your name?")
        for module, node, state in (
            (
                "src.agents.profile.nodes",
                ask_profile_user,
                ProfileState(profile_id=uuid4(), language="en", missing_fields=["name"], next_question=question),
            ),
            (
                "src.agents.resume.nodes",
                ask_resume_user,
                MasterResumeState(profile_id=uuid4(), language="en", missing_fields=["name"], next_question=question),
            ),
        ):
            with self.subTest(module=module), patch(f"{module}.interrupt", return_value="Not sure") as pause:
                update = await cast(Any, node)(state, None)
                expected = {"kind": "question", **question.model_dump()} if module.endswith("profile.nodes") else question.model_dump()
                pause.assert_called_once_with(expected)
                self.assertNotIn("missing_fields", update)
                self.assertEqual(state.missing_fields, ["name"])
                self.assertEqual(update["raw_answers"][0]["text"], question.text)

    async def test_resume_final_answer_is_analyzed_before_saving_in_both_modes(self):
        for mode in ("manual", "profile"):
            with self.subTest(mode=mode):
                question = NextQuestion(field="personalInfo.name", text="What is your name?")
                analysis = AsyncMock(side_effect=[
                    ResumeOutput(resume_data=resume_data(), missing_fields=[question.field], next_question=question),
                    ResumeOutput(resume_data=resume_data("Ada"), missing_fields=[], next_question=None),
                ])
                saved_id = uuid4()
                resume_service = SimpleNamespace(
                    prepare_resume_question=analysis if mode == "manual" else AsyncMock(),
                    master_resume_from_data=Mock(return_value=SimpleNamespace(id=saved_id)),
                    save_resume=AsyncMock(),
                )
                profile_service = SimpleNamespace(
                    analyze_profile_and_prepare_resume_question=analysis if mode == "profile" else AsyncMock()
                )
                context = MasterResumeContext(cast(Any, resume_service), cast(Any, profile_service))
                graph = create_resume_agent(InMemorySaver())
                config: RunnableConfig = {"configurable": {"thread_id": str(uuid4())}}
                result = await graph.ainvoke(
                    MasterResumeState(
                        profile_id=uuid4(), language="en",
                        creation_method=cast(Literal["manual", "profile"], mode),
                    ),
                    config=config, context=context,
                )
                self.assertEqual(result["__interrupt__"][0].value, question.model_dump())
                resume_service.save_resume.assert_not_awaited()
                result = await graph.ainvoke(Command(resume="Ada"), config=config, context=context)
                self.assertEqual(analysis.await_count, 2)
                assert analysis.await_args is not None
                self.assertEqual(analysis.await_args.kwargs["last_answer"], "Ada")
                self.assertEqual(analysis.await_args.kwargs["raw_answers"][0]["answer"], "Ada")
                self.assertEqual(resume_service.master_resume_from_data.call_args.kwargs["data"].personalInfo.name, "Ada")
                self.assertEqual(result["resume_id"], saved_id)
                resume_service.save_resume.assert_awaited_once()

    async def test_complete_resume_skips_question_in_both_modes(self):
        for mode in ("manual", "profile"):
            with self.subTest(mode=mode):
                analysis = AsyncMock(return_value=ResumeOutput(
                    resume_data=resume_data("Ada"), missing_fields=[], next_question=None
                ))
                service = SimpleNamespace(
                    prepare_resume_question=analysis,
                    master_resume_from_data=Mock(return_value=SimpleNamespace(id=uuid4())),
                    save_resume=AsyncMock(),
                )
                context = MasterResumeContext(
                    cast(Any, service),
                    cast(Any, SimpleNamespace(analyze_profile_and_prepare_resume_question=analysis)),
                )
                result = await create_resume_agent(InMemorySaver()).ainvoke(
                    MasterResumeState(
                        profile_id=uuid4(), language="en",
                        creation_method=cast(Literal["manual", "profile"], mode),
                    ),
                    config={"configurable": {"thread_id": str(uuid4())}}, context=context,
                )
                self.assertNotIn("__interrupt__", result)
                service.save_resume.assert_awaited_once()

    async def test_unresolved_answer_is_reasked_instead_of_saving(self):
        question = NextQuestion(field="personalInfo.name", text="What is your name?")
        analysis = AsyncMock(return_value=ResumeOutput(
            resume_data=resume_data(), missing_fields=[question.field], next_question=question
        ))
        service = SimpleNamespace(prepare_resume_question=analysis, save_resume=AsyncMock())
        context = MasterResumeContext(cast(Any, service), cast(Any, SimpleNamespace()))
        graph = create_resume_agent(InMemorySaver())
        config: RunnableConfig = {"configurable": {"thread_id": str(uuid4())}}
        await graph.ainvoke(
            MasterResumeState(profile_id=uuid4(), language="en", creation_method="manual"),
            config=config, context=context,
        )
        result = await graph.ainvoke(Command(resume="Maybe"), config=config, context=context)
        self.assertIn("__interrupt__", result)
        self.assertEqual(result["missing_fields"], [question.field])
        self.assertEqual(analysis.await_count, 2)
        service.save_resume.assert_not_awaited()

    async def test_profile_final_answer_is_incorporated_before_review_and_save(self):
        question = NextQuestion(field="name", text="What is your name?")
        service = SimpleNamespace(
            get_profile_data=AsyncMock(return_value=profile_data("")),
            analyze_data=AsyncMock(side_effect=[
                ProfileOutput(profile_data=profile_data(""), missing_fields=["name"], next_question=question),
                ProfileOutput(profile_data=profile_data(), missing_fields=[], next_question=None),
                ProfileOutput(profile_data=profile_data(), missing_fields=[], next_question=None),
            ]),
            update_profile=AsyncMock(),
        )
        graph = create_profile_agent(InMemorySaver())
        config: RunnableConfig = {"configurable": {"thread_id": str(uuid4())}}
        context = ProfileContext(cast(Any, service))
        await graph.ainvoke(ProfileState(profile_id=uuid4(), language="en"), config=config, context=context)
        result = await graph.ainvoke(Command(resume="Ada"), config=config, context=context)
        self.assertEqual(result["__interrupt__"][0].value["profile"]["name"], "Ada")
        service.update_profile.assert_not_awaited()
        # A rejected review must not save or silently count as approval.
        await graph.ainvoke(Command(resume={"approved": False}), config=config, context=context)
        service.update_profile.assert_not_awaited()
        await graph.ainvoke(Command(resume={"approved": True}), config=config, context=context)
        self.assertEqual(service.analyze_data.await_count, 3)
        self.assertEqual(service.update_profile.await_args.kwargs["data"].name, "Ada")
        service.update_profile.assert_awaited_once()

    async def test_invalid_creation_choice_prompts_again_and_can_be_corrected(self):
        service = SimpleNamespace(
            prepare_resume_question=AsyncMock(return_value=ResumeOutput(
                resume_data=resume_data("Ada"), missing_fields=[], next_question=None
            )),
            master_resume_from_data=Mock(return_value=SimpleNamespace(id=uuid4())),
            save_resume=AsyncMock(),
        )
        graph = create_resume_agent(InMemorySaver())
        config: RunnableConfig = {"configurable": {"thread_id": str(uuid4())}}
        context = MasterResumeContext(cast(Any, service), cast(Any, SimpleNamespace()))
        await graph.ainvoke(
            MasterResumeState(profile_id=uuid4(), language="en"),
            config=config, context=context,
        )
        result = await graph.ainvoke(Command(resume="invalid"), config=config, context=context)
        self.assertIn("error", result["__interrupt__"][0].value)
        service.prepare_resume_question.assert_not_awaited()
        result = await graph.ainvoke(Command(resume="manual"), config=config, context=context)
        self.assertNotIn("__interrupt__", result)
        service.save_resume.assert_awaited_once()

    async def test_review_edits_are_reanalyzed_and_need_fresh_approval(self):
        question = NextQuestion(field="name", text="What is your name?")
        service = SimpleNamespace(
            get_profile_data=AsyncMock(return_value=profile_data()),
            analyze_data=AsyncMock(side_effect=[
                ProfileOutput(profile_data=profile_data(), missing_fields=[], next_question=None),
                ProfileOutput(profile_data=profile_data(""), missing_fields=["name"], next_question=question),
                ProfileOutput(profile_data=profile_data("Grace"), missing_fields=[], next_question=None),
            ]),
            update_profile=AsyncMock(),
        )
        graph = create_profile_agent(InMemorySaver())
        config: RunnableConfig = {"configurable": {"thread_id": str(uuid4())}}
        context = ProfileContext(cast(Any, service))
        await graph.ainvoke(
            ProfileState(profile_id=uuid4(), language="en"), config=config, context=context
        )
        edited = profile_data("").model_dump()
        edited["raw_data"] = "This must not replace the original story"
        result = await graph.ainvoke(
            Command(resume={"profile": edited, "approved": True}), config=config, context=context
        )
        self.assertEqual(result["__interrupt__"][0].value, {"kind": "question", **question.model_dump()})
        self.assertEqual(service.analyze_data.await_args.kwargs["data"].raw_data, "Original story")
        service.update_profile.assert_not_awaited()
        result = await graph.ainvoke(Command(resume="Grace"), config=config, context=context)
        self.assertEqual(result["__interrupt__"][0].value["profile"]["name"], "Grace")
        service.update_profile.assert_not_awaited()
        # Invalid review input should produce a new, correctable interrupt.
        result = await graph.ainvoke(Command(resume="invalid"), config=config, context=context)
        self.assertIn("error", result["__interrupt__"][0].value)
        await graph.ainvoke(Command(resume={"approved": True}), config=config, context=context)
        self.assertEqual(service.update_profile.await_args.kwargs["data"].name, "Grace")
        service.update_profile.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
