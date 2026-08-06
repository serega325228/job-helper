from uuid import UUID


class ProfileNotFoundError(LookupError):
    def __init__(self, profile_id: UUID) -> None:
        self.profile_id = profile_id
        super().__init__(f"Profile {profile_id} was not found")
