import unittest
from unittest.mock import patch
from uuid import uuid4

from src.infrastructure.models.preference_intent import PreferenceIntent
from src.infrastructure.models.profile import Profile
from src.infrastructure.models.vacancy import Vacancy
from src.schemas.vacancy import VacancySoftConditions
from src.services.embedding_text import build_vacancy_search_text
from src.services.scoring import ScoringService
from src.services.skill_canonicalization import SkillCanonicalizer


class FakeReranker:
    model_name = "fake-reranker"

    async def score(self, query: str, documents: list[str]) -> list[float]:
        return [0.8 for _ in documents]


class FakeEmbeddingService:
    model_name = "fake-embedding"

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, float(index)] for index, _ in enumerate(texts)]


async def run_inline(function, *args):
    return function(*args)


def make_profile() -> Profile:
    return Profile(
        id=uuid4(),
        name="Candidate",
        raw_story="Backend developer",
        profile_summary="Backend engineer with production experience",
        skills=["Golang", "Postgres"],
        experience=["Four years of backend development"],
        education=[],
        seniority="middle",
        experience_years=4,
        contacts={},
    )


def make_vacancy() -> Vacancy:
    soft = VacancySoftConditions(
        summary="Backend services and distributed systems",
        required_skills=["Go", "PostgreSQL", "Python"],
        preferred_skills=["Kubernetes"],
        responsibilities=["Develop backend services"],
        industries=["fintech"],
    )
    return Vacancy(
        id=uuid4(),
        source="hh",
        external_id="42",
        url="https://hh.ru/vacancy/42",
        title="Senior Backend Developer",
        company_name="Example",
        description="Full vacancy description",
        city="Москва",
        work_format="remote",
        employment_type="full_time",
        experience="between1And3",
        seniority="middle",
        salary_from=200_000,
        salary_to=300_000,
        salary_currency="RUR",
        soft_conditions=soft.model_dump(mode="json"),
        raw_payload={},
        content_embedding=[1.0, 0.0],
        title_embedding=[1.0, 0.0],
    )


def make_preference(profile_id) -> PreferenceIntent:
    return PreferenceIntent(
        id=uuid4(),
        profile_id=profile_id,
        name="Backend",
        description="Backend product development",
        target_titles=["Backend Developer"],
        required_skills=["Postgres", "Go"],
        preferred_skills=["K8s"],
        preferred_industries=["fintech"],
        preferred_companies=["Example"],
        excluded_titles=[],
        excluded_companies=[],
        locations=["Москва"],
        work_formats=["remote"],
        employment_types=["full_time"],
        salary_min=180_000,
        salary_currency="RUR",
        min_seniority="junior",
        max_seniority="senior",
        content_embedding=[1.0, 0.0],
        title_embedding=[1.0, 0.0],
        weight=1.0,
        enabled=True,
    )


class ScoringServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.service = ScoringService(
            FakeReranker(),
            FakeEmbeddingService(),
            SkillCanonicalizer(),
        )

    def test_vacancy_embedding_text_contains_only_semantic_fields(self) -> None:
        vacancy = make_vacancy()

        text = build_vacancy_search_text(vacancy)

        self.assertIn("Backend services", text)
        self.assertIn("PostgreSQL", text)
        self.assertIn("Develop backend services", text)
        self.assertNotIn("300000", text)
        self.assertNotIn("remote", text)
        self.assertNotIn("Москва", text)
        self.assertNotIn("middle", text)

    def test_profile_comparison_canonicalizes_skill_aliases(self) -> None:
        comparison = self.service.compare_profile(make_profile(), make_vacancy())

        self.assertIn("postgresql", comparison.matched_skills)
        self.assertIn("go", comparison.matched_skills)
        self.assertEqual(comparison.missing_skills, ["python"])
        self.assertGreater(comparison.score, 0.6)

    def test_preference_comparison_uses_structured_conditions(self) -> None:
        profile = make_profile()
        comparison = self.service.compare_preference(
            make_preference(profile.id),
            make_vacancy(),
        )

        self.assertTrue(comparison.hard_constraints_passed)
        self.assertEqual(comparison.components["required_skills"], 1.0)
        self.assertEqual(comparison.components["location"], 1.0)
        self.assertGreater(comparison.score, 0.8)

    async def test_reranks_profile_and_preference_separately(self) -> None:
        profile = make_profile()
        vacancy = make_vacancy()
        preference = make_preference(profile.id)

        results = await self.service.rerank_vacancies(
            profile,
            [vacancy],
            {vacancy.id: preference},
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[vacancy.id].profile_score, 0.8)
        self.assertEqual(results[vacancy.id].preference_score, 0.8)

    async def test_preference_embedding_builder_sets_both_vectors(self) -> None:
        profile = make_profile()
        preference = make_preference(profile.id)
        preference.content_embedding = None
        preference.title_embedding = None

        with patch("src.services.scoring.asyncio.to_thread", new=run_inline):
            await self.service.update_preference_embeddings([preference])

        self.assertEqual(preference.title_embedding, [1.0, 0.0])
        self.assertEqual(preference.content_embedding, [1.0, 1.0])
        self.assertEqual(preference.embedding_model, "fake-embedding")


if __name__ == "__main__":
    unittest.main()
