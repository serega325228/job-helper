import threading
import unittest
from datetime import UTC, datetime
from unittest.mock import Mock

from src.exceptions.vacancy import VacancyPreviewEvaluationError
from src.infrastructure.laya.laya_provider import LayaProvider
from src.infrastructure.laya.questions import MATCH_QUESTIONS
from src.infrastructure.models.preference_intent import PreferenceIntent
from src.infrastructure.models.profile import Profile
from src.schemas.vacancy import (
    VacancyHardFilters,
    VacancyPreview,
    VacancyScrapingQuery,
    WorkFormat,
)
from src.services.vacancy_preview import VacancyPreviewEvaluator


def preview(**values) -> VacancyPreview:
    return VacancyPreview.model_validate(
        {
            "source": "site",
            "external_id": "1",
            "url": "https://example.com/jobs/1",
            "title": "Python developer",
            **values,
        },
    )


def preference(**values) -> PreferenceIntent:
    return PreferenceIntent(
        **{
            "name": "Python",
            "enabled": True,
            "target_titles": ["Python developer"],
            "excluded_titles": [],
            "excluded_companies": [],
            "work_formats": [],
            "employment_types": [],
            "locations": [],
            "required_skills": [],
            "preferred_skills": [],
            **values,
        },
    )


def result(role="good", skill="weak") -> dict:
    return {
        "answers": {"role_fit": {"choice": role}, "skill_fit": {"choice": skill}},
    }


class VacancyPreviewTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.laya = Mock(spec=LayaProvider)
        self.evaluator = VacancyPreviewEvaluator(self.laya)
        self.profile = Profile(
            skills=["Python"],
            experience=["Built APIs"],
            experience_years=3,
        )
        self.query = VacancyScrapingQuery(text="Python")

    def accept(self, vacancy, preferences=None, **filters):
        return self.evaluator.accept(
            vacancy,
            self.profile,
            preferences or [],
            VacancyHardFilters(**filters),
            self.query,
        )

    def test_unknown_values_survive_explicit_filters(self):
        self.assertTrue(
            self.accept(
                preview(),
                cities=["London"],
                work_formats=["remote"],
                salary_min=100,
                salary_currency="USD",
                salary_gross=False,
                seniorities=["junior"],
            ),
        )
        self.assertTrue(
            self.accept(
                preview(title="Senior developer"),
                seniorities=["junior", "staff"],
            ),
        )
        self.assertFalse(
            self.accept(preview(title="Senior developer"), seniorities=["junior"]),
        )

    def test_preferences_are_alternatives_and_disabled_preferences_do_not_qualify(self):
        excluded = preference(excluded_titles=["Python"])
        allowed = preference()
        self.assertTrue(self.accept(preview(), [excluded, allowed]))
        self.assertFalse(self.accept(preview(), [excluded]))
        self.assertFalse(self.accept(preview(), [preference(enabled=False)]))
        self.assertFalse(
            self.accept(
                preview(title="Senior Python developer"),
                [preference(max_seniority="middle")],
            ),
        )

    def test_keyword_boundaries_preserve_language_names(self):
        self.query = VacancyScrapingQuery(text="developer", required_keywords=["Java"])
        self.assertFalse(self.accept(preview(title="JavaScript developer")))
        self.assertTrue(self.accept(preview(title="Java developer")))
        self.query = VacancyScrapingQuery(text="developer", excluded_keywords=["Go"])
        self.assertTrue(
            self.accept(preview(company_name="Google", title="Google developer")),
        )
        self.assertFalse(self.accept(preview(title="Go developer")))
        self.query = VacancyScrapingQuery(text="developer", required_keywords=["C++"])
        self.assertTrue(self.accept(preview(title="C++ developer")))
        self.assertFalse(self.accept(preview(title="C++17 developer")))

    def test_query_and_hard_filter_publication_cutoffs_both_apply(self):
        self.query = VacancyScrapingQuery(
            text="Python", published_after=datetime(2026, 10, 1, tzinfo=UTC)
        )
        filters = {"published_after": datetime(2026, 9, 1, tzinfo=UTC)}
        self.assertFalse(
            self.accept(
                preview(published_at=datetime(2026, 9, 15, tzinfo=UTC)), **filters
            )
        )
        self.assertTrue(self.accept(preview(), **filters))
        self.assertTrue(
            self.accept(
                preview(published_at=datetime(2026, 10, 2, tzinfo=UTC)), **filters
            )
        )

    def test_only_explicit_experience_minimums_reject(self):
        self.assertFalse(self.accept(preview(experience="5+ years of experience")))
        self.assertTrue(self.accept(preview(experience="between3And6")))
        self.assertFalse(self.accept(preview(experience="moreThan6")))
        self.assertTrue(self.accept(preview(experience="3–5 years")))
        self.assertTrue(self.accept(preview(experience="up to 5 years")))
        self.assertTrue(self.accept(preview(experience="Python 3 experience")))

    def test_salary_requires_known_comparable_ceiling_currency_and_gross(self):
        filters = {"salary_min": 100, "salary_currency": "USD", "salary_gross": False}
        self.assertFalse(
            self.accept(
                preview(salary_to=50, salary_currency="USD", salary_gross=False),
                **filters,
            ),
        )
        self.assertTrue(
            self.accept(
                preview(salary_to=50, salary_currency="EUR", salary_gross=False),
                **filters,
            ),
        )
        self.assertTrue(
            self.accept(preview(salary_to=50, salary_currency="USD"), **filters),
        )
        self.assertTrue(
            self.accept(
                preview(salary_from=50, salary_currency="USD", salary_gross=False),
                **filters,
            ),
        )

    def test_remote_preference_preserves_location_unknowns_but_explicit_geo_filters_apply(
        self,
    ):
        vacancy = preview(location="Berlin", work_format=WorkFormat.REMOTE)
        preferences = [preference(locations=["London"])]
        self.assertTrue(self.accept(vacancy, preferences))
        self.assertTrue(self.accept(preview(location="Berlin"), preferences))
        self.assertFalse(
            self.accept(
                preview(location="Berlin", work_format=WorkFormat.ON_SITE), preferences
            )
        )
        self.assertFalse(self.accept(vacancy, preferences, cities=["London"]))
        self.assertFalse(
            self.accept(preview(location="New York"), cities=["Yorkshire"])
        )

    async def test_evaluation_batches_use_existing_questions_and_leave_event_loop_thread(
        self,
    ):
        calling_thread = threading.get_ident()
        evaluation_threads = []

        def evaluate(states, questions, *, batch_size, **kwargs):
            evaluation_threads.append(threading.get_ident())
            self.assertIs(questions, MATCH_QUESTIONS)
            self.assertEqual(batch_size, 2)
            self.assertEqual(states[0]["candidate"]["skills"], ["Python"])
            return [result("good", "weak"), result("weak", "strong")][: len(states)]

        provider = LayaProvider()
        provider._agent = Mock()
        provider._agent.predict_batch.side_effect = evaluate
        self.evaluator = VacancyPreviewEvaluator(provider)
        vacancies = [preview(external_id=str(index)) for index in range(3)]
        accepted = await self.evaluator.evaluate(
            vacancies,
            self.profile,
            [preference()],
            self.query,
            batch_size=2,
        )
        self.assertEqual(accepted, [vacancies[0], vacancies[2]])
        self.assertEqual(provider._agent.predict_batch.call_count, 2)
        self.assertTrue(all(thread != calling_thread for thread in evaluation_threads))

    async def test_malformed_or_incomplete_evaluation_is_rejected(self):
        for results in ([], [result("invented", "good")], [{}]):
            with self.subTest(results=results):
                self.laya.evaluate_batch.return_value = results
                with self.assertRaises(VacancyPreviewEvaluationError):
                    await self.evaluator.evaluate(
                        [preview()],
                        self.profile,
                        [],
                        self.query,
                        batch_size=2,
                    )
        self.laya.evaluate_batch.return_value = [result("strong", "none")]
        self.assertEqual(
            await self.evaluator.evaluate(
                [preview()],
                self.profile,
                [],
                self.query,
                batch_size=2,
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
