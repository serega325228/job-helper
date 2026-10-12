from pathlib import Path
from typing import Annotated
from uuid import UUID

from dishka import FromDishka
from dishka.integrations.fastapi import inject
from fastapi import File, UploadFile
from fastapi.exceptions import HTTPException
from fastapi.routing import APIRouter
from structlog import get_logger

from src.services.resume import ResumeService

logger = get_logger()

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

router = APIRouter(prefix="resume", tags=["Resume"])


@router.post("/import-master-resume/{profile_id}", status_code=201)
@inject
async def import_master_resume(
    file: Annotated[UploadFile, File()],
    profile_id: Annotated[UUID, Path()],
    resume_service: FromDishka[ResumeService]
):
    logger.info("Uploading master resume")

    suffix = Path(file.filename or "").suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail="Only PDF, DOCX, and DOC files are allowed",
        )

    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="File size exceeds the 20 MB limit",
        )

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Empty file",
        )

    try:
        await resume_service.import_master_resume(
            profile_id=profile_id,
            suffix=suffix,
            content=content,
        )


    except ValueError as e:
        raise HTTPException(
            status_code=500,
            detail="Failed to parse resume",
        ) from e
