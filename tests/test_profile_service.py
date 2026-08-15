import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from src.services.profile import ProfileService


class FakeUnitOfWork:
    def __init__(self) -> None:
        self.profiles = AsyncMock()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        return None


class ProfileServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_gets_preferences_through_repository(self) -> None:
        expected = [object()]
        unit_of_work = FakeUnitOfWork()
        unit_of_work.profiles.get_preferences_by_profile_id.return_value = expected
        service = ProfileService(unit_of_work, AsyncMock())
        profile_id = uuid4()

        result = await service.get_preferences_by_profile_id(profile_id)

        self.assertIs(result, expected)
        unit_of_work.profiles.get_preferences_by_profile_id.assert_awaited_once_with(
            profile_id,
        )


if __name__ == "__main__":
    unittest.main()
