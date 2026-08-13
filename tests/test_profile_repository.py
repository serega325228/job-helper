import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from sqlalchemy.dialects import postgresql

from src.repositories.profile import ProfileRepository


class ProfileRepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_gets_preferences_by_profile_id(self) -> None:
        expected = object()
        session = AsyncMock()
        session.scalars.return_value = [expected]
        repository = ProfileRepository(session)
        profile_id = uuid4()

        result = await repository.get_preferences_by_profile_id(profile_id)

        self.assertEqual(result, [expected])
        statement = session.scalars.await_args.args[0]
        compiled = statement.compile(dialect=postgresql.dialect())
        sql = str(compiled)
        self.assertIn("preference_intents.profile_id =", sql)
        self.assertIn(
            "ORDER BY preference_intents.created_at, preference_intents.id",
            sql,
        )
        self.assertEqual(compiled.params["profile_id_1"], profile_id)


if __name__ == "__main__":
    unittest.main()
