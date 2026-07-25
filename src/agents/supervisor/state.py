from typing import Literal

from pydantic.main import BaseModel


class AgentResult(BaseModel):
    status: Literal["success", "needs_input", "failed", "partial"]
    summary: str
    artifact_ids: list[str] = []
    suggested_next_action: str | None = None
    warnings: list[str] = []
    metrics: dict[str, int | float | str] = {}
