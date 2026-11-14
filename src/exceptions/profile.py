from uuid import UUID


class ProfileNotFoundError(LookupError):
    def __init__(self, profile_id: UUID) -> None:
        self.profile_id = profile_id
        super().__init__(f"Profile {profile_id} was not found")

class ProfileWorkflowError(Exception):
    def __init__(self, code: str, status_code: int) -> None:
        self.code = code
        self.status_code = status_code
        super().__init__(code)


class ProfileTurnError(Exception):
    def __init__(self, profile_id: UUID, thread_id: UUID, error: Exception) -> None:
        self.profile_id = profile_id
        self.thread_id = thread_id
        self.error = error
        super().__init__("Profile wizard turn failed")


class AnalysisError(Exception):
    def __init__(self, e: Exception) -> None:
        self.e = e
        super().__init__(f"Analysis error: {e}")
