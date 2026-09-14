from uuid import UUID

from infrastructure.db.sqlalchemy_unit_of_work import SqlAlchemyUnitOfWork
from infrastructure.models.resume import Resume, ResumeType


class ResumeService:
    def __init__(
        self,
        unit_of_work: SqlAlchemyUnitOfWork,
    ):
        self._uow = unit_of_work

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
