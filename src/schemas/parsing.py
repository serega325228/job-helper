from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field


class VacancyDocument(BaseModel):
    url: str
    title_hint: str | None = None
    clean_text: str = Field(max_length=30_000)
    structured_data: dict[str, Any] | None = None
    extraction_source: Literal["json_ld", "html"]


@dataclass(slots=True)
class DownloadedPage:
    url: str
    html: str
    rendered: bool
