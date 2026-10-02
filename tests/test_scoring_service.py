import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

from src.infrastructure.laya.laya_provider import LayaProvider
from src.infrastructure.models.preference_intent import PreferenceIntent
from src.infrastructure.models.profile import Profile
from src.infrastructure.models.vacancy import Vacancy
from src.schemas.vacancy import VacancySoftConditions
from src.services.embedding_text import (
    build_preference_title_text,
    build_vacancy_search_text,
)
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
        self.laya = Mock(spec=LayaProvider)
        self.service = ScoringService(
            FakeReranker(),
            FakeEmbeddingService(),
            SkillCanonicalizer(),
            self.laya,
        )

    async def test_laya_pairs_each_vacancy_with_its_preference(self) -> None:
        profile = make_profile()
        first_preference = make_preference(profile.id)
        second_preference = make_preference(profile.id)
        second_preference.name = "Platform"
        second_preference.target_titles = ["Platform Engineer"]
        vacancies = [make_vacancy() for _ in range(3)]
        preferences_by_vacancy = {
            vacancies[2].id: first_preference,
            vacancies[0].id: first_preference,
            vacancies[1].id: second_preference,
        }
        self.laya.evaluate_batch.return_value = [
            {
                "answers": {
                    "role_fit": {"choice": role_fit},
                    "skill_fit": {"choice": "strong"},
                },
            }
            for role_fit in ("good", "weak", "strong")
        ]

        result = await self.service.laya_match_vacancies(
            profile,
            preferences_by_vacancy,
            {vacancy.id: vacancy for vacancy in vacancies},
        )

        self.assertEqual(list(result), [vacancy.id for vacancy in vacancies])
        self.assertEqual(result[vacancies[0].id].role_fit, "good")
        self.assertEqual(result[vacancies[1].id].role_fit, "weak")
        self.assertEqual(result[vacancies[2].id].role_fit, "strong")
        states = self.laya.evaluate_batch.call_args.args[0]
        self.assertEqual(
            [state["candidate"]["target_role"] for state in states],
            [
                build_preference_title_text(first_preference),
                build_preference_title_text(second_preference),
                build_preference_title_text(first_preference),
            ],
        )

    async def test_laya_skips_empty_input_and_rejects_missing_preferences(self) -> None:
        profile = make_profile()
        self.assertEqual(await self.service.laya_match_vacancies(profile, {}, {}), {})
        vacancy = make_vacancy()

        with self.assertRaisesRegex(ValueError, "Preferences are missing"):
            await self.service.laya_match_vacancies(profile, {}, {vacancy.id: vacancy})

        self.laya.evaluate_batch.assert_not_called()

    async def test_laya_rejects_incomplete_batch_results(self) -> None:
        profile = make_profile()
        vacancy = make_vacancy()
        self.laya.evaluate_batch.return_value = []

        with self.assertRaises(ValueError):
            await self.service.laya_match_vacancies(
                profile,
                {vacancy.id: make_preference(profile.id)},
                {vacancy.id: vacancy},
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


if __name__ == "__main__":
    unittest.main()
