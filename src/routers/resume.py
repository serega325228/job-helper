from pathlib import Path
from typing import Annotated
from uuid import UUID
import json


from dishka import FromDishka
from dishka.integrations.fastapi import inject
from fastapi import File, UploadFile
from fastapi.exceptions import HTTPException
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRouter
from langgraph.types import Command
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
            filename=suffix,
            content=content,
        )


    except ValueError as e:
        raise HTTPException(
            status_code=500,
            detail="Failed to parse resume",
        ) from e

    logger.info("Master resume imported successfully")

# Move to wizard router
async def event_stream(
    thread_id: str,
    user_input: str,
):
    config = {"configurable": {"thread_id": thread_id}}
    if user_input:
        graph_input = Command(resume=user_input)
    else:
        graph_input = {}

    async for chunck in graph.astream(
        graph_input,
        config=config,
        stream_node="messages",
        version="v2",
    ):
        if chunck["type"] == "messages":
            yield f"data: {json.dumps({"type": "token", "content": chunck["data"]})}\n\n"
        elif chunck["type"] == "updates":
            for node_name, state in chunck["data"].items():
                yield f"data: {json.dumps({"type": "node_update", "node": node_name, "state": state})}\n\n"

        if "__interrupt__" in chunck.get("data", {}):
            interrupt_data = chunck["data"]["__interrupt__"][0]
            yield f"data: {json.dumps({"type": "interrupt", "data": interrupt_data.value})}\n\n"

    yield "data: [DONE]\n\n"

@router.post("/wizard/stream")
async def stream_wizard(thread_id: str, answer: str = ""):
    return StreamingResponse(
        event_stream(thread_id, answer),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Prevents NGINX from buffering the stream
        },
    )
