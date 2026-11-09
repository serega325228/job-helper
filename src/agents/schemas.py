"""Shared contract for one analysis/question turn in an interactive agent."""

from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class NextQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: NonEmptyText
    field: NonEmptyText = Field(description="The exact path in missing_fields[0].")


class FollowUpOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Required: an omitted list must never silently mean that analysis is done.
    missing_fields: list[NonEmptyText] = Field(
        description="Unresolved, relevant field paths in priority order; [] when done."
    )
    next_question: NextQuestion | None = Field(
        description="One question about missing_fields[0], or null when the list is empty."
    )

    @model_validator(mode="after")
    def validate_follow_up(self) -> Self:
        if not self.missing_fields:
            if self.next_question is not None:
                raise ValueError("next_question must be null when missing_fields is empty")
        elif self.next_question is None:
            raise ValueError("next_question is required when missing_fields is nonempty")
        elif self.next_question.field != self.missing_fields[0]:
            raise ValueError("next_question.field must match missing_fields[0]")
        return self
