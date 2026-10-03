import logging
import re
from datetime import UTC, datetime
from urllib.parse import urlencode, urljoin, urlsplit

from playwright.async_api import Page
from selectolax.lexbor import LexborHTMLParser, LexborNode
from structlog import get_logger

from src.exceptions.vacancy import VacancyScrapingError
from src.schemas.vacancy import (
    EmploymentType,
    RawVacancy,
    VacancyHardFilters,
    VacancyPreview,
    VacancyScrapingQuery,
    WorkFormat,
)
from src.services.parsing import ParsingService

logger = get_logger()

RESULTS_SELECTOR = ".jobs-search__results-list"
CARD_SELECTOR = f"{RESULTS_SELECTOR} .base-card"
EMPTY_SELECTOR = ".no-results, .jobs-search-no-results-banner, .jobs-search__no-results"
END_SELECTOR = ".see-more-jobs__viewed-all"
JOB_PATH_PATTERN = re.compile(r"/jobs/view/(?:[^/]*-)?([0-9]+)/?$")
WORK_FORMAT_CODES = {
    WorkFormat.ON_SITE: "1",
    WorkFormat.REMOTE: "2",
    WorkFormat.HYBRID: "3",
}
EMPLOYMENT_CODES = {
    EmploymentType.FULL_TIME: "F",
    EmploymentType.PART_TIME: "P",
    EmploymentType.CONTRACT: "C",
    EmploymentType.TEMPORARY: "T",
    EmploymentType.INTERNSHIP: "I",
}
SENIORITY_CODES = {
    "intern": "1",
    "internship": "1",
    "entry level": "2",
    "junior": "2",
    "associate": "3",
    "middle": "4",
    "senior": "4",
    "mid-senior level": "4",
    "director": "5",
    "executive": "6",
}


class LinkedInVacancySource:
    source_name = "linkedin"

    def search_url(
        self,
        query: VacancyScrapingQuery,
        filters: VacancyHardFilters,
    ) -> str:
        params = {"keywords": query.text.strip()}
        location = query.location
        if not location and len(filters.cities) == 1:
            location = filters.cities[0]
        if not location and len(filters.countries) == 1:
            location = filters.countries[0]
        if location:
            params["location"] = location

        for name, values, codes in (
            ("f_WT", filters.work_formats, WORK_FORMAT_CODES),
            ("f_JT", filters.employment_types, EMPLOYMENT_CODES),
            (
                "f_E",
                [value.casefold().replace("_", " ") for value in filters.seniorities],
                SENIORITY_CODES,
            ),
        ):
            if values and all(value in codes for value in values):
                params[name] = ",".join(dict.fromkeys(codes[value] for value in values))

        return f"https://www.linkedin.com/jobs/search/?{urlencode(params)}"

    async def apply_filters(
        self,
        page: Page,
        query: VacancyScrapingQuery,
        filters: VacancyHardFilters,
    ) -> None:
        self._validate_page_url(page.url)
        await page.locator(f"{RESULTS_SELECTOR}, {EMPTY_SELECTOR}").first.wait_for(
            state="attached",
        )

    async def parse_previews(self, page: Page) -> list[VacancyPreview]:
        self._validate_page_url(page.url)
        tree = LexborHTMLParser(await page.content())
        cards = tree.css(CARD_SELECTOR)
        if not cards:
            if tree.css_first(EMPTY_SELECTOR) is not None:
                return []
            raise VacancyScrapingError(
                "LinkedIn search has no recognizable vacancy cards"
            )

        previews: list[VacancyPreview] = []
        for card in cards:
            link = card.css_first(".base-card__full-link[href]")
            href = link.attrs.get("href") if link is not None else None
            try:
                if not href:
                    raise ValueError("Vacancy card is missing its URL")
                external_id, url = self._vacancy_url(href, page.url)
                title = self._text(card, ".base-search-card__title")
                if not title:
                    raise ValueError("Vacancy card is missing its title")
                published = card.css_first("time[datetime]")
                previews.append(
                    VacancyPreview(
                        source=self.source_name,
                        external_id=external_id,
                        url=url,
                        title=title,
                        company_name=self._text(card, ".base-search-card__subtitle"),
                        location=self._text(card, ".job-search-card__location"),
                        short_description=self._text(card, ".job-search-card__snippet"),
                        salary=self._text(card, ".job-search-card__salary-info"),
                        published_at=self._published_at(
                            published.attrs.get("datetime")
                            if published is not None
                            else None,
                        ),
                    ),
                )
            except (ValueError, VacancyScrapingError) as error:
                logger.warning(
                    "Malformed vacancy preview source=%s url=%s request_type=SEARCH: %r",
                    self.source_name,
                    href or page.url,
                    error,
                )

        if not previews:
            raise VacancyScrapingError("All LinkedIn vacancy cards were malformed")
        return previews

    async def advance_search(self, page: Page) -> bool:
        self._validate_page_url(page.url)
        before = len(LexborHTMLParser(await page.content()).css(CARD_SELECTOR))
        end = page.locator(END_SELECTOR)
        if await end.is_visible():
            return False
        button = page.get_by_role("button", name="See more jobs", exact=True)
        if await button.count() == 0:
            return False

        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await page.wait_for_function(
            """({cards, count, end}) =>
                document.querySelectorAll(cards).length > count ||
                Array.from(document.querySelectorAll(end)).some(node => node.getClientRects().length) ||
                Array.from(document.querySelectorAll('button.infinite-scroller__show-more-button'))
                    .some(node => node.getClientRects().length && !node.disabled)
            """,
            arg={"cards": CARD_SELECTOR, "count": before, "end": END_SELECTOR},
        )
        if len(LexborHTMLParser(await page.content()).css(CARD_SELECTOR)) > before:
            return True
        if await end.is_visible():
            return False

        await button.click()
        await page.wait_for_function(
            """({cards, count, end}) =>
                document.querySelectorAll(cards).length > count ||
                Array.from(document.querySelectorAll(end)).some(node => node.getClientRects().length)
            """,
            arg={"cards": CARD_SELECTOR, "count": before, "end": END_SELECTOR},
        )
        self._validate_page_url(page.url)
        return len(LexborHTMLParser(await page.content()).css(CARD_SELECTOR)) > before

    async def parse_vacancy(self, page: Page, preview: VacancyPreview) -> RawVacancy:
        self._validate_page_url(page.url)
        external_id, url = self._vacancy_url(page.url, page.url)
        if preview.source != self.source_name or external_id != preview.external_id:
            raise VacancyScrapingError("LinkedIn detail URL does not match its preview")
        await page.locator(
            '.description__text, script[type="application/ld+json"]',
        ).first.wait_for(state="attached")
        html = await page.content()
        tree = LexborHTMLParser(html)
        posting = self._job_posting(html, page.url, external_id)
        description = tree.css_first(".description__text .show-more-less-html__markup")
        raw_text = ParsingService.html_to_clean_text(
            description.html
            if description is not None
            else str(posting.get("description") or ""),
        )
        if not raw_text:
            raise VacancyScrapingError(
                "LinkedIn vacancy description is missing or empty"
            )
        criteria: dict[str, str] = {}
        for item in tree.css(".description__job-criteria-item"):
            name = self._text(item, ".description__job-criteria-subheader")
            value = self._text(item, ".description__job-criteria-text")
            if name and value:
                criteria[name] = value

        top_card = tree.css_first(".top-card-layout")
        date_node = (
            top_card.css_first("time[datetime]") if top_card is not None else None
        )
        published_at = (
            self._published_at(
                posting.get("datePosted")
                or (date_node.attrs.get("datetime") if date_node is not None else None),
            )
            or preview.published_at
        )
        payload = dict(posting)
        payload.update(
            preview=preview.model_dump(mode="json"),
            company_name=self._text(tree, ".topcard__org-name-link")
            or preview.company_name,
            location=self._text(tree, ".topcard__flavor-row .topcard__flavor--bullet")
            or preview.location,
            criteria=criteria,
        )
        return RawVacancy(
            source=self.source_name,
            external_id=external_id,
            url=url,
            title=self._text(tree, ".top-card-layout__title")
            or posting.get("title")
            or preview.title,
            raw_text=raw_text,
            raw_payload=payload,
            published_at=published_at,
            fetched_at=datetime.now(UTC),
        )

    @staticmethod
    def _text(tree: LexborHTMLParser | LexborNode, selector: str) -> str | None:
        node = tree.css_first(selector)
        if node is None:
            return None
        return (
            ParsingService.normalize_text(node.text(separator=" ", strip=True)) or None
        )

    @staticmethod
    def _published_at(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            published_at = datetime.fromisoformat(value)
        except TypeError, ValueError:
            return None
        return (
            published_at
            if published_at.tzinfo is not None
            else published_at.replace(tzinfo=UTC)
        )

    @staticmethod
    def _validate_page_url(url: str) -> None:
        parts = urlsplit(url)
        if (
            parts.scheme != "https"
            or not parts.hostname
            or not (
                parts.hostname == "linkedin.com"
                or parts.hostname.endswith(".linkedin.com")
            )
            or parts.username is not None
            or parts.password is not None
            or parts.port not in (None, 443)
            or not parts.path.startswith("/jobs/")
        ):
            raise VacancyScrapingError("LinkedIn page redirected outside public jobs")

    @classmethod
    def _vacancy_url(cls, value: str, base_url: str) -> tuple[str, str]:
        url = urljoin(base_url, value)
        cls._validate_page_url(url)
        match = JOB_PATH_PATTERN.fullmatch(urlsplit(url).path)
        if match is None:
            raise ValueError("LinkedIn vacancy URL must contain a numeric job ID")
        external_id = match.group(1)
        return external_id, f"https://www.linkedin.com/jobs/view/{external_id}/"

    @classmethod
    def _job_posting(cls, html: str, page_url: str, external_id: str) -> dict:
        postings = ParsingService.extract_job_postings(html)
        for posting in postings:
            if not posting.get("url"):
                if len(postings) == 1:
                    return posting
                continue
            try:
                posting_id, _ = cls._vacancy_url(str(posting["url"]), page_url)
            except ValueError, VacancyScrapingError:
                continue
            if posting_id == external_id:
                return posting
        return {}
