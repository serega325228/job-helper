import unittest
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import parse_qs, urlsplit

from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from src.exceptions.vacancy import VacancyScrapingError
from src.infrastructure.vacancy_sources.linkedin.source import LinkedInVacancySource
from src.schemas.vacancy import VacancyHardFilters, VacancyPreview, VacancyScrapingQuery
from src.services.parsing import ParsingService

SEARCH_HTML = """
<ul class="jobs-search__results-list">
  <li><div class="base-card" data-entity-urn="urn:li:jobPosting:42">
    <a class="base-card__full-link" href="https://uk.linkedin.com/jobs/view/backend-python-at-example-42?trackingId=abc#apply"></a>
    <h3 class="base-search-card__title"> Backend   Python Engineer </h3>
    <h4 class="base-search-card__subtitle"> Example Company </h4>
    <span class="job-search-card__location"> London, United Kingdom </span>
    <span class="job-search-card__salary-info"> £50,000 - £60,000 </span>
    <time datetime="2026-09-30">2 days ago</time>
  </div></li>
</ul>
"""

DETAIL_HTML = """
<section class="top-card-layout">
  <h1 class="top-card-layout__title">Backend Python Engineer</h1>
  <div class="topcard__flavor-row">
    <a class="topcard__org-name-link">Example Company</a>
    <span class="topcard__flavor--bullet">London, United Kingdom</span>
  </div>
  <span class="topcard__flavor--bullet">200 applicants</span>
</section>
<section class="description__text">
  <div class="show-more-less-html__markup"><p>Build Python services.</p><p>Use PostgreSQL.</p><script>ignore me</script></div>
  <button aria-label="Show more">Show more</button>
</section>
<ul><li class="description__job-criteria-item">
  <h3 class="description__job-criteria-subheader">Employment type</h3>
  <span class="description__job-criteria-text">Full-time</span>
</li></ul>
<aside><time datetime="2020-01-01">Related vacancy date</time></aside>
"""


def fake_page(html: str, url: str = "https://www.linkedin.com/jobs/search/"):
    page = MagicMock()
    page.url = url
    page.content = AsyncMock(return_value=html)
    page.locator.return_value.first.wait_for = AsyncMock()
    page.locator.return_value.is_visible = AsyncMock(return_value=False)
    page.get_by_role.return_value.count = AsyncMock(return_value=1)
    page.get_by_role.return_value.click = AsyncMock()
    page.evaluate = AsyncMock()
    page.wait_for_function = AsyncMock()
    return page


class LinkedInVacancySourceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.source = LinkedInVacancySource()
        self.preview = VacancyPreview(
            source="linkedin",
            external_id="42",
            title="Backend Python Engineer",
            url="https://www.linkedin.com/jobs/view/42/",
        )

    def test_native_filters_and_url_encoding(self) -> None:
        url = self.source.search_url(
            VacancyScrapingQuery(text="C++ & Python", location="London, UK"),
            VacancyHardFilters(
                work_formats=["remote", "hybrid"],
                employment_types=["full_time", "contract"],
                seniorities=["junior", "senior"],
            ),
        )

        self.assertEqual(
            parse_qs(urlsplit(url).query),
            {
                "keywords": ["C++ & Python"],
                "location": ["London, UK"],
                "f_WT": ["2,3"],
                "f_JT": ["F,C"],
                "f_E": ["2,4"],
            },
        )

    def test_unsupported_native_choices_do_not_exclude_allowed_choices(self) -> None:
        url = self.source.search_url(
            VacancyScrapingQuery(text="Python"),
            VacancyHardFilters(
                work_formats=["remote", "other"], cities=["London", "Paris"]
            ),
        )
        self.assertEqual(parse_qs(urlsplit(url).query), {"keywords": ["Python"]})

    async def test_preview_bulk_parsing_and_canonical_identity(self) -> None:
        page = fake_page(SEARCH_HTML)
        previews = await self.source.parse_previews(page)

        self.assertEqual(len(previews), 1)
        preview = previews[0]
        self.assertEqual(preview.external_id, "42")
        self.assertEqual(str(preview.url), "https://www.linkedin.com/jobs/view/42/")
        self.assertEqual(preview.title, "Backend Python Engineer")
        self.assertEqual(preview.company_name, "Example Company")
        self.assertEqual(preview.location, "London, United Kingdom")
        self.assertEqual(preview.salary, "£50,000 - £60,000")
        self.assertEqual(preview.published_at, datetime(2026, 9, 30, tzinfo=UTC))
        self.assertIsNone(preview.work_format)
        self.assertIsNone(preview.experience)
        page.locator.assert_not_called()

    async def test_malformed_individual_cards_are_skipped(self) -> None:
        for href in (
            "https://linkedin.com.evil.example/jobs/view/55/",
            "http://www.linkedin.com/jobs/view/55/",
            "https://www.linkedin.com/jobs/view/not-an-id/",
            "https://username:password@www.linkedin.com/jobs/view/55/",
            "https://www.linkedin.com:444/jobs/view/55/",
        ):
            with self.subTest(href=href):
                html = SEARCH_HTML.replace(
                    "</ul>",
                    f'<li><div class="base-card"><a class="base-card__full-link" href="{href}"></a><h3 class="base-search-card__title">Bad card</h3></div></li></ul>',
                )
                with self.assertLogs(
                    "src.infrastructure.vacancy_sources.linkedin.source",
                    level="WARNING",
                ):
                    previews = await self.source.parse_previews(fake_page(html))
                self.assertEqual([preview.external_id for preview in previews], ["42"])

    async def test_empty_results_and_blocked_pages_are_distinguished(self) -> None:
        result = await self.source.parse_previews(
            fake_page(
                '<section class="core-section-container no-results">No jobs found</section>'
            ),
        )
        self.assertEqual(result, [])
        for html, url in (
            ("<h1>Sign in</h1>", "https://www.linkedin.com/authwall"),
            ("<h1>Security check</h1>", "https://www.linkedin.com/jobs/search/"),
            (
                '<ul class="jobs-search__results-list"><li>Changed markup</li></ul>',
                "https://www.linkedin.com/jobs/search/",
            ),
        ):
            with self.subTest(url=url), self.assertRaises(VacancyScrapingError):
                await self.source.parse_previews(fake_page(html, url))

    async def test_apply_filters_waits_for_rendered_listing(self) -> None:
        page = fake_page(SEARCH_HTML)
        await self.source.apply_filters(
            page,
            VacancyScrapingQuery(text="Python"),
            VacancyHardFilters(),
        )
        page.locator.return_value.first.wait_for.assert_awaited_once_with(
            state="attached"
        )
        page.goto.assert_not_called()

    async def test_load_more_waits_for_new_cards(self) -> None:
        page = fake_page(SEARCH_HTML)
        page.content.side_effect = [
            SEARCH_HTML,
            SEARCH_HTML,
            SEARCH_HTML.replace("</ul>", '<li><div class="base-card"></div></li></ul>'),
        ]

        self.assertTrue(await self.source.advance_search(page))
        page.get_by_role.assert_called_once_with(
            "button", name="See more jobs", exact=True
        )
        page.get_by_role.return_value.click.assert_awaited_once()
        self.assertEqual(page.wait_for_function.await_count, 2)

    async def test_native_scroll_growth_does_not_click_another_page(self) -> None:
        page = fake_page(SEARCH_HTML)
        page.content.side_effect = [
            SEARCH_HTML,
            SEARCH_HTML.replace("</ul>", '<li><div class="base-card"></div></li></ul>'),
        ]

        self.assertTrue(await self.source.advance_search(page))
        page.get_by_role.return_value.click.assert_not_awaited()

    async def test_end_state_stops_and_load_timeout_propagates(self) -> None:
        page = fake_page(SEARCH_HTML)
        page.locator.return_value.is_visible.return_value = True
        self.assertFalse(await self.source.advance_search(page))
        page.evaluate.assert_not_awaited()

        page = fake_page(SEARCH_HTML)
        page.wait_for_function.side_effect = PlaywrightTimeoutError("load-more timeout")
        with self.assertRaises(PlaywrightTimeoutError):
            await self.source.advance_search(page)

    async def test_detail_text_and_metadata_preserve_source_facts(self) -> None:
        raw = await self.source.parse_vacancy(
            fake_page(
                DETAIL_HTML,
                "https://uk.linkedin.com/jobs/view/backend-python-at-example-42?trackingId=abc",
            ),
            self.preview,
        )

        self.assertEqual(raw.raw_text, "Build Python services.\nUse PostgreSQL.")
        self.assertEqual(raw.raw_payload["criteria"], {"Employment type": "Full-time"})
        self.assertEqual(raw.raw_payload["location"], "London, United Kingdom")
        self.assertIsNone(raw.published_at)
        self.assertNotIn("ignore me", raw.raw_text)

    async def test_nested_structured_data_reuses_existing_parsing_helpers(self) -> None:
        html = """<script type="application/ld+json">{
            "@graph": [{"@type": ["Thing", "JobPosting"], "title": "Python Engineer",
                "url": "https://www.linkedin.com/jobs/view/42/", "datePosted": "2026-10-01T09:00:00Z",
                "description": "<p>Build services.</p><p>Use Python.</p>"}]
        }</script><script type="application/ld+json">invalid json</script>"""
        self.assertEqual(len(ParsingService.extract_job_postings(html)), 1)
        raw = await self.source.parse_vacancy(
            fake_page(html, "https://www.linkedin.com/jobs/view/42/"),
            self.preview,
        )

        self.assertEqual(raw.title, "Python Engineer")
        self.assertEqual(raw.raw_text, "Build services.\nUse Python.")
        self.assertEqual(raw.published_at, datetime(2026, 10, 1, 9, tzinfo=UTC))

    async def test_removed_or_mismatched_details_fail(self) -> None:
        for html, url in (
            (
                "<h1>This vacancy is no longer available</h1>",
                "https://www.linkedin.com/jobs/view/42/",
            ),
            (DETAIL_HTML, "https://www.linkedin.com/jobs/view/55/"),
        ):
            with self.subTest(url=url), self.assertRaises(VacancyScrapingError):
                await self.source.parse_vacancy(fake_page(html, url), self.preview)


if __name__ == "__main__":
    unittest.main()
