from src.infrastructure.models.profile import Profile
from src.repositories.profile import ProfileRepository


class ProfileService:
    def __init__(self, profile_repository: ProfileRepository):
        self._profiles = profile_repository

    async def create_profile(
        self,
        name: str,
        raw_story: str,
    ):
        profile = Profile(name=name, raw_story=raw_story)
        return await self._profiles.create(profile)
