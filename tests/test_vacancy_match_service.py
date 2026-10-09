import math
import unittest
from uuid import uuid4

from src.config.settings import Settings
from src.schemas.scoring import LayaComparison, PreferenceComparison, ProfileComparison
from src.schemas.vacancy_match import MatchCategory
from src.services.vacancy_match import VacancyMatchService


class VacancyMatchServiceTests(unittest.TestCase):
    def test_builds_final_result_with_geometric_score(self) -> None:
        service = VacancyMatchService(None, None, Settings(_env_file=None))
        result = service.build_result(
            profile_id=uuid4(),
            vacancy_id=uuid4(),
            preference_intent_id=uuid4(),
            profile_comparison=ProfileComparison(
                score=0.7,
                matched_skills=["python"],
                missing_skills=["redis"],
            ),
            preference_comparison=PreferenceComparison(
                score=0.8,
                hard_constraints_passed=True,
            ),
            laya_comparison=LayaComparison(role_fit="good", skill_fit="strong"),
            profile_rerank_score=0.9,
            preference_rerank_score=0.85,
            title_similarity=0.95,
            content_similarity=0.75,
            embedding_similarity=0.83,
        )

        expected_score = math.prod(
            (0.7**0.16, 0.8**0.16, 0.9**0.24, 0.85**0.24, (2 / 3) ** 0.1),
        )
        self.assertAlmostEqual(result.total_score, expected_score)
        self.assertEqual(result.category, MatchCategory.TARGET)
        self.assertEqual(result.matched_skills, ["python"])
        self.assertEqual(result.missing_skills, ["redis"])
        self.assertEqual(result.component_scores["semantic"]["combined"], 0.83)
        self.assertEqual(result.component_scores["laya"]["role_score"], 2 / 3)
        self.assertEqual(result.component_scores["laya"]["skill_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
