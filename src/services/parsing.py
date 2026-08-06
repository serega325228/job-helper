import json
import re
from collections.abc import Iterator
from typing import Any
from urllib.parse import urldefrag, urljoin

import httpx
from playwright.async_api import Browser
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from selectolax.lexbor import LexborHTMLParser

from src.schemas.parsing import DownloadedPage, VacancyDocument

UNWANTED_SELECTORS = """
script,
style,
noscript,
svg,
canvas,
iframe,
nav,
header,
footer,
form,
button,
aside
"""

WORD_PATTERN = re.compile(r"[a-zа-яё0-9+#.]+", re.IGNORECASE)

JOB_URL_PATTERN = re.compile(
    r"/(?:jobs?|vacanc(?:y|ies)|careers?|positions?|openings?)(?:/|[-_?=])",
    re.IGNORECASE,
)

def visible_text_length(html: str) -> int:
    tree = LexborHTMLParser(html)

    for node in tree.css("script, style, noscript"):
        node.decompose()

    if tree.body is None:
        return 0

    return len(tree.body.text(strip=True))


class ParsingService:
    def __init__(
        self,
        http_client: httpx.AsyncClient,
        browser: Browser,
    ) -> None:
        self._http = http_client
        self._browser = browser

    async def fetch(
        self,
        url: str,
        render_js: bool | None = None,
    ) -> DownloadedPage:
        if render_js is not True:
            try:
                response = await self._http.get(url)
                response.raise_for_status()

                content_type = response.headers.get("content-type", "")

                if (
                    "text/html" in content_type
                    and visible_text_length(response.text) >= 300
                ):
                    return DownloadedPage(
                        url=str(response.url),
                        html=response.text,
                        rendered=False,
                    )
            except httpx.HTTPError:
                if render_js is False:
                    raise

        return await self._fetch_with_browser(url)

    async def _fetch_with_browser(self, url: str) -> DownloadedPage:
        context = await self._browser.new_context()
        page = await context.new_page()

        async def handle_route(route) -> None:
            if route.request.resource_type in {
                "image",
                "media",
                "font",
            }:
                await route.abort()
            else:
                await route.continue_()

        await page.route("**/*", handle_route)

        try:
            await page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=30_000,
            )

            try:
                await page.wait_for_function(
                    "document.body && document.body.innerText.length > 300",
                    timeout=10_000,
                )
            except PlaywrightTimeoutError:
                pass

            return DownloadedPage(
                url=page.url,
                html=await page.content(),
                rendered=True,
            )
        finally:
            await context.close()

    def walk_json(self, value: Any) -> Iterator[dict]:
        if isinstance(value, dict):
            item_type = value.get("@type")
            types = item_type if isinstance(item_type, list) else [item_type]

            if "JobPosting" in types:
                yield value

            for nested_value in value.values():
                yield from self.walk_json(nested_value)

        elif isinstance(value, list):
            for item in value:
                yield from self.walk_json(item)


    def extract_job_postings(self, html: str) -> list[dict]:
        tree = LexborHTMLParser(html)
        result: list[dict] = []

        for script in tree.css('script[type="application/ld+json"]'):
            raw_json = script.text(deep=True, strip=False).strip()

            if not raw_json:
                continue

            try:
                data = json.loads(raw_json)
            except json.JSONDecodeError:
                continue

            result.extend(self.walk_json(data))

        return result

    def normalize_text(self, text: str) -> str:
        result: list[str] = []
        previous_line: str | None = None

        for raw_line in text.splitlines():
            line = re.sub(r"\s+", " ", raw_line).strip()

            if not line:
                continue

            if line == previous_line:
                continue

            result.append(line)
            previous_line = line

        return "\n".join(result)


    def html_to_clean_text(self, html: str) -> str:
        tree = LexborHTMLParser(html)

        for node in tree.css(UNWANTED_SELECTORS):
            node.decompose()

        content = None

        for selector in (
            "main",
            "article",
            '[role="main"]',
            ".job-description",
            ".vacancy-description",
        ):
            content = tree.css_first(selector)

            if content is not None:
                break

        content = content or tree.body

        if content is None:
            return ""

        text = content.text(
            separator="\n",
            strip=True,
        )

        return self.normalize_text(text)

    def words(self, value: str) -> set[str]:
        return {
            word.casefold()
            for word in WORD_PATTERN.findall(value)
        }


    def title_may_match(
        self,
        text: str,
        target_titles: list[str],
    ) -> bool:
        text_words = self.words(text)

        return any(
            self.words(target_title) <= text_words
            for target_title in target_titles
        )


    def find_candidate_links(
        self,
        html: str,
        base_url: str,
        target_titles: list[str],
    ) -> list[str]:
        tree = LexborHTMLParser(html)
        result: list[str] = []
        seen: set[str] = set()

        for link in tree.css("a[href]"):
            href = link.attrs.get("href")

            if not href:
                continue

            url = urljoin(base_url, href)
            url, _ = urldefrag(url)

            context_node = link.parent or link
            context_text = context_node.text(
                separator=" ",
                strip=True,
            )[:500]

            looks_like_job_url = bool(JOB_URL_PATTERN.search(url))
            title_matches = self.title_may_match(
                context_text,
                target_titles,
            )

            if looks_like_job_url and title_matches and url not in seen:
                seen.add(url)
                result.append(url)

        return result

    def parse_vacancy_page(
        self,
        url: str,
        html: str,
    ) -> VacancyDocument:
        postings = self.extract_job_postings(html)

        if postings:
            posting = postings[0]

            description = posting.get("description") or ""

            return VacancyDocument(
                url=url,
                title_hint=posting.get("title") or posting.get("name"),
                clean_text=self.html_to_clean_text(description),
                structured_data=posting,
                extraction_source="json_ld",
            )

        tree = LexborHTMLParser(html)
        h1 = tree.css_first("h1")

        return VacancyDocument(
            url=url,
            title_hint=h1.text(strip=True) if h1 else None,
            clean_text=self.html_to_clean_text(html)[:30_000],
            structured_data=None,
            extraction_source="html",
        )
