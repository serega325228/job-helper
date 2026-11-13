import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID

from docling.exceptions import ConversionError
from sqlalchemy.exc import IntegrityError

from src.agents.resume.schemas import ResumeOutput
from src.agents.schemas import NextQuestion
from src.exceptions.resume import ResumeError
from src.infrastructure.db.sqlalchemy_unit_of_work import SqlAlchemyUnitOfWork
from src.infrastructure.document_parser.document_parser import DocumentParser
from src.infrastructure.llm.llm import LLMProvider
from src.infrastructure.models.resume import Resume, ResumeStatus, ResumeType
from src.prompts.resume_wizard import RESUME_WIZARD_TURN_PROMPT
from src.repositories.prompt import PromptRepository
from src.schemas.resume import ResumeData


class ResumeService:
    def __init__(
        self,
        unit_of_work: SqlAlchemyUnitOfWork,
        document_parser: DocumentParser,
        prompts: PromptRepository,
        llm: LLMProvider,
    ):
        self._uow = unit_of_work
        self._document_parser = document_parser
        self._prompts = prompts
        self._llm = llm

    async def ensure_profile(
        self, profile_id: UUID, *, creating: bool = False, resume_id: UUID | None = None
    ) -> None:
        async with self._uow as uow:
            if await uow.profiles.get_by_id(profile_id) is None:
                raise ResumeError("profile_not_found", 404)
            if creating:
                existing = await uow.resumes.get_master(profile_id)
                if existing is not None and existing.id != resume_id:
                    raise ResumeError("master_resume_exists", 409)

    async def get_master(self, profile_id: UUID, resume_id: UUID | None = None) -> Resume:
        async with self._uow as uow:
            if await uow.profiles.get_by_id(profile_id) is None:
                raise ResumeError("profile_not_found", 404)
            resume = await uow.resumes.get_master(profile_id, resume_id)
            if resume is None:
                raise ResumeError("master_resume_not_found", 404)
            return resume

    async def import_master_resume(
        self,
        profile_id: UUID,
        filename: str,
        content: bytes,
        language: str,
    ) -> Resume:
        await self.ensure_profile(profile_id, creating=True)
        try:
            with TemporaryDirectory() as temp_dir:
                # Never use the supplied path for filesystem access.
                path = Path(temp_dir) / f"resume{Path(filename).suffix.lower()}"
                await asyncio.to_thread(path.write_bytes, content)
                extracted = await asyncio.to_thread(self._document_parser.parse, path)
        except ConversionError:
            raise ResumeError("document_not_extractable", 422) from None
        except Exception:
            raise ResumeError("document_parser_unavailable", 503) from None
        if not extracted.strip():
            raise ResumeError("document_text_empty", 422)
        parsed = await self._llm.complete(
            prompt=f"Extract the factual resume in {language}. Candidate's resume: {extracted}",
            schema=ResumeData,
        )
        resume = self.master_resume_from_data(
            profile_id=profile_id,
            filename=filename,
            title=parsed.personalInfo.title or "",
            language=language,
            raw_data=extracted,
            data=parsed,
        )
        return await self.save_resume(resume)

    async def save_resume(self, resume: Resume) -> Resume:
        try:
            async with self._uow as uow:
                # Every master writer locks the parent, including imports and graph saves.
                if await uow.profiles.get_by_id(resume.profile_id, lock=True) is None:
                    raise ResumeError("profile_not_found", 404)
                existing = await uow.resumes.get_master(resume.profile_id)
                if existing is not None:
                    # A checkpoint save-node replay may only recover its own server-issued ID.
                    if resume.id is not None and existing.id == resume.id:
                        return existing
                    raise ResumeError("master_resume_exists", 409)
                await uow.resumes.create(resume)
        except IntegrityError as error:
            constraint = getattr(getattr(error.orig, "__cause__", None), "constraint_name", None)
            if constraint == "uq_resumes_profile_master_resume":
                raise ResumeError("master_resume_exists", 409) from None
            if constraint == "resumes_profile_id_fkey":
                raise ResumeError("profile_not_found", 404) from None
            raise ResumeError("resume_storage_unavailable", 503) from None
        return resume

    async def prepare_resume_question(
        self,
        language: str,
        last_question: NextQuestion | None,
        last_answer: str,
        data: ResumeData,
        *,
        raw_answers: list[dict] | None = None,
    ) -> ResumeOutput:
        template, _ = self._prompts.get(
            "resume_wizard_turn_prompt", RESUME_WIZARD_TURN_PROMPT
        )
        return await self._llm.complete(
            json.dumps(
                {
                    "resume_data": data.model_dump(mode="json"),
                    "last_question": last_question.model_dump() if last_question else None,
                    "answer_text": last_answer,
                    "answer_history": raw_answers or [],
                },
                ensure_ascii=False,
            ),
            system_prompt=template.format(output_language=language),
            schema=ResumeOutput,
        )

    @staticmethod
    def master_resume_from_data(
        profile_id: UUID,
        title: str,
        language: str,
        data: ResumeData,
        filename: str | None = None,
        raw_data: str | None = None,
    ) -> Resume:
        return Resume(
            profile_id=profile_id,
            type=ResumeType.MASTER,
            filename=filename,
            title=title,
            raw_data=raw_data or "",
            language=language,
            processed_data=data.model_dump(mode="json"),
            status=ResumeStatus.COMPLETED,
        )
