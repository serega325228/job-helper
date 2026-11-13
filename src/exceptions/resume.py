from uuid import UUID


class ResumeError(Exception):
    """Stable public errors; underlying provider/database details stay private."""

    def __init__(self, code: str, status_code: int) -> None:
        super().__init__(code)
        self.code = code
        self.status_code = status_code


class ResumeTurnError(Exception):
    """Preserve an initial turn's recovery handle without publishing its cause."""

    def __init__(self, thread_id: UUID, error: Exception) -> None:
        super().__init__("resume_turn_failed")
        self.thread_id = thread_id
        self.error = error
