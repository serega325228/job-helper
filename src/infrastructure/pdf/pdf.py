import asyncio
import logging
import os
import sys
from pathlib import Path

from config.settings import PDFSettings
from playwright.async_api import (
    Browser,
    Error as PlaywrightError,
    Page,
    Playwright,
)

logger = logging.getLogger(__name__)

# Explicit, bounded navigation/selector timeout. Chosen over Playwright's
# implicit 30s default so a slow-but-working render (large resume, cold cache,
# modest hardware) still completes, while a genuinely stuck page still fails
# in finite time rather than hanging.
_NAV_TIMEOUT_MS = 60_000


class PDFRenderError(Exception):
    """Custom exception for PDF rendering errors with helpful messages."""
    pass

class PDFRender:
    def __init__(self, playwright: Playwright, settings: PDFSettings):
        self._playwright = playwright
        self._browser: Browser | None = None
        self._browser_lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(settings.max_concurrency)

    @staticmethod
    def _resolve_pdf_format(page_size: str) -> str:
        format_map = {
            "A4": "A4",
            "LETTER": "Letter",
        }
        return format_map.get(page_size, "A4")

    @staticmethod
    def _resolve_pdf_margins(margins: dict | None) -> dict:
        if margins:
            return {
                "top": f"{margins.get('top', 10)}mm",
                "right": f"{margins.get('right', 10)}mm",
                "bottom": f"{margins.get('bottom', 10)}mm",
                "left": f"{margins.get('left', 10)}mm",
            }
        return {"top": "10mm", "right": "10mm", "bottom": "10mm", "left": "10mm"}

    @staticmethod
    def _find_chromium_executable() -> str | None:
        """Find system Chrome/Chromium/Edge executable across platforms."""
        if sys.platform == "win32":
            candidates = [
                Path(os.environ.get("PROGRAMFILES", "C:/Program Files"))
                / "Google/Chrome/Application/chrome.exe",
                Path(os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)"))
                / "Google/Chrome/Application/chrome.exe",
                Path(os.environ.get("PROGRAMFILES", "C:/Program Files"))
                / "Microsoft/Edge/Application/msedge.exe",
                Path(os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)"))
                / "Microsoft/Edge/Application/msedge.exe",
            ]
        elif sys.platform == "darwin":
            # macOS application paths
            candidates = [
                Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
                Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
                Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
            ]
        else:
            # Linux paths: standard locations, Snap, and Flatpak
            candidates = [
                Path("/usr/bin/google-chrome"),
                Path("/usr/bin/google-chrome-stable"),
                Path("/usr/bin/chromium"),
                Path("/usr/bin/chromium-browser"),
                Path("/usr/bin/microsoft-edge"),
                Path("/snap/bin/chromium"),
                Path("/var/lib/flatpak/exports/bin/com.google.Chrome"),
                Path("/var/lib/flatpak/exports/bin/org.chromium.Chromium"),
                Path(os.path.expanduser("~/.local/share/flatpak/exports/bin/com.google.Chrome")),
                Path(os.path.expanduser("~/.local/share/flatpak/exports/bin/org.chromium.Chromium")),
            ]

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return None


    async def _get_browser(self) -> Browser:
        if self._browser is not None and self._browser.is_connected():
            return self._browser
        async with self._browser_lock:
            if self._browser is not None and self._browser.is_connected():
                return self._browser
            try:
                self._browser = await self._playwright.chromium.launch(headless=True)
                return self._browser
            except PlaywrightError as e:
                if "Executable doesn't exist" not in str(e):
                    raise
                fallback_executable = self._find_chromium_executable()
                if not fallback_executable:
                    raise PDFRenderError(
                        "Playwright browser executable is missing, and no system Chrome/Edge "
                        "installation was found. Install Playwright browsers or install Chrome/Edge."
                    ) from e
                self._browser = await self._playwright.chromium.launch(executable_path=fallback_executable, headless=True)
                return self._browser

    @staticmethod
    async def _render_page_to_pdf(
        page: Page,
        url: str,
        selector: str,
        pdf_format: str,
        pdf_margins: dict,
    ) -> bytes:
        # NOTE: do NOT use wait_until="networkidle" here. The Next.js dev server
        # (HMR/Turbopack + RSC streaming) keeps the network busy, so "idle" may
        # never arrive and goto silently hangs until timeout → 503 (issues
        # #799/#808), with the failure depending on environment/network noise.
        # Wait on the real readiness condition instead — document "load", the
        # resume content selector, and fonts — all bounded by an explicit timeout
        # so the outcome is deterministic.
        await page.goto(url, wait_until="load", timeout=_NAV_TIMEOUT_MS)
        await page.wait_for_selector(selector, timeout=_NAV_TIMEOUT_MS)
        # Bound the fonts wait too — plain page.evaluate has no timeout, so a stuck
        # font load could otherwise hang the render past _NAV_TIMEOUT_MS.
        await page.wait_for_function(
            "() => document.fonts.ready.then(() => true)", timeout=_NAV_TIMEOUT_MS
        )
        return await page.pdf(
            format=pdf_format,
            print_background=True,
            margin=pdf_margins,
        )

    # @staticmethod
    # def _raise_playwright_error(error: PlaywrightError, url: str) -> NoReturn:
    #     error_msg = str(error)
    #     if "Executable doesn't exist" in error_msg:
    #         exe = sys.executable.replace("\\", "/")
    #         command = f"{exe} -m playwright install chromium"
    #         raise PDFRenderError(
    #             "Playwright browser executable is missing or out of date. "
    #             "Command shown for reference; quote the path if it contains spaces: "
    #             f"{command}"
    #         ) from error
    #     if "net::ERR_CONNECTION_REFUSED" in error_msg:
    #         raise PDFRenderError(
    #             f"Cannot connect to frontend for PDF generation. "
    #             f"Attempted URL: {url}. "
    #             f"Please ensure: 1) The frontend is running, "
    #             f"2) The FRONTEND_BASE_URL environment variable in the backend .env file "
    #             f"matches the URL where your frontend is accessible."
    #         ) from error
    #     # Catch-all: the raw Playwright message can carry internal navigation URLs
    #     # and a full call log. Log it server-side; return a generic message to the
    #     # client (CLAUDE.md rule 5 — and it stops the verbose trace from overflowing
    #     # the client error modal, #811).
    #     logger.error("PDF rendering failed for %s: %s", url, error_msg)
    #     raise PDFRenderError(
    #         "PDF rendering failed. Please try again, or try a simpler resume or a "
    #         "different template."
    #     ) from error

    async def close(self) -> None:
        """Close the Playwright browser instance."""
        async with self._browser_lock:
            if self._browser is None:
                return

            await self._browser.close()
            self._browser = None


    async def render_resume_pdf(
        self,
        url: str,
        page_size: str = "A4",
        selector: str = ".resume-print",
        margins: dict | None = None,
    ) -> bytes:
        """Render a URL to PDF bytes.

        Args:
            url: The URL to render (print route)
            page_size: Page size format - "A4" or "LETTER"
            selector: CSS selector to wait for before rendering (default: ".resume-print")
            margins: Page margins dict with top/right/bottom/left in mm (applied to every page)

        Note:
            Margins are applied via Playwright's PDF margins, ensuring they appear
            on every page (not just the first page like HTML padding would).
        """
        async with self._semaphore:
            pdf_format = self._resolve_pdf_format(page_size)
            pdf_margins = self._resolve_pdf_margins(margins)

            browser = await self._get_browser()

            ctx = await browser.new_context()

            try:
                page = await ctx.new_page()
                try:
                    return await self._render_page_to_pdf(page, url, selector, pdf_format, pdf_margins)
                finally:
                    await page.close()
            finally:
                await ctx.close()
