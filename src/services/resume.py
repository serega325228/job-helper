import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID

from fastapi.exceptions import HTTPException

from infrastructure.db.sqlalchemy_unit_of_work import SqlAlchemyUnitOfWork
from src.infrastructure.document_parser.document_parser import DocumentParser
from src.infrastructure.llm.llm import LLMProvider
from src.infrastructure.models.resume import Resume, ResumeStatus, ResumeType
from src.schemas.resume import ResumeData


class ResumeService:
    def __init__(
        self,
        unit_of_work: SqlAlchemyUnitOfWork,
        document_parser: DocumentParser,
        llm: LLMProvider,
    ):
        self._uow = unit_of_work
        self._document_parser = document_parser
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
        suffix: str,
        content: bytes,
    ) -> Resume:
        try:
            extracted = await self._exctract_resume_data(suffix, content)

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
        resume = Resume(
            profile_id=profile_id,
            resume_type=ResumeType.MASTER,
            content=content,
            filename=suffix,
            title=parsed.personal_info.title,
            raw_data=extracted,
            language=parsed.content_language,
            processed_data=parsed.model_dump(mode="json"),
            status=ResumeStatus.COMPLETED,
        )

        async with self._uow as uow:
            await uow.resumes.create(resume)

        return resume

    async def create_base_resume(
        self,
        profile_id: UUID,
        name: str,
        language: str,
        content: dict,
    ) -> Resume:
        resume = Resume(
            profile_id=profile_id,
            name=name,
            language=language,
            resume_type=ResumeType.BASE,
            content=content,
        )
        async with self._uow as uow:
            await uow.resumes.create(resume)

        return resume
