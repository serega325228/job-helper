from datetime import datetime

from langgraph.graph.message import uuid
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql.functions import func
from sqlalchemy.types import DateTime, Text, String

from src.infrastructure.models.base import Base


class VacancyModel(Base):
    __tablename__ = "vacancies"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)

    source: Mapped[str] = mapped_column(
        String(50),
        index=True
    )

    title: Mapped[str] = mapped_column(String(500))
    company: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
