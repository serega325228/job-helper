import asyncio
from fileinput import filename
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID

from fastapi.exceptions import HTTPException

from src.agents.resume.schemas import WizardTurnOutput
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

    async def _exctract_resume_data(self, filename: str, content: bytes) -> str:
        try:
            with TemporaryDirectory() as temp_dir:
                path = Path(temp_dir) / f"resume_{filename}"
                path.write_bytes(content)
                return await asyncio.to_thread(self._document_parser.parse, path)
        except Exception as e:
            raise ValueError(f"Failed to parse resume: {e}") from e

    async def _parse_resume(self, content: str) -> ResumeData:
        try:
            parsed = await self._llm.complete(
                prompt=f"Candidate's resume: {content}",
                schema=ResumeData,
            )
            return parsed
        except Exception as e:
            raise ValueError(f"Failed to parse resume: {e}") from e

    async def import_master_resume(
        self,
        profile_id: UUID,
        filename: str,
        content: bytes,
    ) -> Resume:
        try:
            extracted = await self._exctract_resume_data(filename, content)

            if not extracted.strip():
                raise HTTPException(
                    status_code=422,
                    detail="No extractable text found in resume",
                )

            parsed = await self._parse_resume(extracted)

        except Exception as e:  # not http maybe
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse resume: {e}",
            ) from e

        # Maybe in try block
        resume = self.master_resume_from_data(
            profile_id=profile_id,
            filename=filename,
            title=parsed.personal_info.title or "",
            language=parsed.content_language, # Should think about language
            raw_data=extracted,
            data=parsed,
        )

        await self.save_resume(resume)

        return resume

    async def save_resume(
        self,
        resume: Resume,
    ) -> Resume:
        async with self._uow as uow:
            await uow.resumes.create(resume)
        return resume

    async def prepare_resume_question(
        self,
        language: str,
        current_section: str,
        last_answer: str,
        data: ResumeData,
    ) -> WizardTurnOutput:
        prompt = self._prompts.get("custom_key", RESUME_WIZARD_TURN_PROMPT) # Get the fucking rid of prompt repo
        prompt = prompt[0].format(
            output_language=language,
            current_section=current_section,
            resume_json=data.model_dump_json(),
            answer_text=last_answer,
        )
        return await self._llm.complete(prompt, schema=WizardTurnOutput)

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
            resume_type=ResumeType.MASTER,
            filename=filename,
            title=title,
            raw_data=raw_data, # DON'T DELETE THIS FUCKING RAW_DATA
            language=language,
            processed_data=data.model_dump(mode="json"),
            status=ResumeStatus.COMPLETED,
        )
