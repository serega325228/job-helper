from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Health check response."""

    status: str


class StatusResponse(BaseModel):
    """Application status response."""

    status: str
    llm_configured: bool
    llm_healthy: bool
    has_master_resume: bool
    database_stats: dict[str, Any]
