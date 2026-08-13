import unittest

from pydantic import ValidationError

from src.schemas.scoring import PreferenceComparison, ProfileComparison


class ProfileComparisonTests(unittest.TestCase):
    def test_validates_score_range(self) -> None:
        with self.assertRaises(ValidationError):
            ProfileComparison(score=1.1)

    def test_uses_independent_collection_defaults(self) -> None:
        first = ProfileComparison(score=0.5)
        second = ProfileComparison(score=0.5)

        first.matched_skills.append("python")

        self.assertEqual(second.matched_skills, [])


class PreferenceComparisonTests(unittest.TestCase):
    def test_validates_score_range(self) -> None:
        with self.assertRaises(ValidationError):
            PreferenceComparison(score=-0.1, hard_constraints_passed=False)


if __name__ == "__main__":
    unittest.main()
