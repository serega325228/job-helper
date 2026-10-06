import unittest
from datetime import UTC, datetime

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


class VacancyPreviewTest(unittest.TestCase):
    def setUp(self):
        self.evaluator = VacancyPreviewEvaluator()
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
                preview(title="Senior Python developer"),
                seniorities=["junior", "staff"],
            ),
        )
        self.assertFalse(
            self.accept(
                preview(title="Senior Python developer"), seniorities=["junior"]
            ),
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
        self.query = VacancyScrapingQuery(text="Java", required_keywords=["Java"])
        self.assertFalse(self.accept(preview(title="JavaScript developer")))
        self.assertTrue(self.accept(preview(title="Java developer")))
        self.query = VacancyScrapingQuery(text="developer", excluded_keywords=["Go"])
        self.assertTrue(
            self.accept(preview(company_name="Google", title="Google developer")),
        )
        self.assertFalse(self.accept(preview(title="Go developer")))
        self.query = VacancyScrapingQuery(text="C++", required_keywords=["C++"])
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

    def test_salary_rejects_known_contradictions_but_preserves_unknown_ceiling(self):
        filters = {"salary_min": 100, "salary_currency": "USD", "salary_gross": False}
        self.assertFalse(
            self.accept(
                preview(salary_to=50, salary_currency="USD", salary_gross=False),
                **filters,
            ),
        )
        self.assertFalse(
            self.accept(
                preview(salary_to=50, salary_currency="EUR", salary_gross=False),
                **filters,
            ),
        )
        self.assertFalse(
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

    def test_partial_ai_preview_survives_but_explicit_contradictions_do_not(self):
        preferences = [
            preference(
                target_titles=["Senior AI engineer"],
                min_seniority="senior",
                salary_min=300_000,
            )
        ]
        for values, accepted in (
            ({"title": "AI developer"}, True),
            ({"title": "Middle AI engineer", "salary_to": 150_000}, False),
            ({"title": "Senior AI engineer", "salary_to": 150_000}, False),
            ({"title": "AI engineer", "salary_from": 150_000}, True),
            (
                {"title": "Senior civil engineer", "short_description": "AI tools"},
                False,
            ),
        ):
            with self.subTest(values=values):
                self.assertEqual(self.accept(preview(**values), preferences), accepted)


if __name__ == "__main__":
    unittest.main()
