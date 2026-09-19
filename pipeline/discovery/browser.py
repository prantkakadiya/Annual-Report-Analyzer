"""Headless browser PDF download via Playwright + direct HTTP fallback."""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from urllib.parse import urlparse
from typing import Callable

import httpx
from playwright.async_api import async_playwright

from pipeline.schema import SearchCandidate

logger = logging.getLogger(__name__)



class ReportDownloader:
    """Downloads annual report PDFs from candidate URLs.

    For candidates flagged as direct PDF links, a simple ``httpx`` GET is
    used.  For HTML pages (common on investor-relations sites), a headless
    Chromium instance renders the page, detects auto-downloads, and clicks
    PDF links.
    """

    def __init__(self, download_dir: str = "data/downloads") -> None:
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Download directory: %s", self.download_dir)

    async def download_from_candidates(
        self, candidates: list[SearchCandidate], company_name: str, progress_callback: Callable[[int, int], None] | None = None
    ) -> list[tuple[SearchCandidate, str]]:
        """Try each candidate in score order, return successful downloads.

        Stops after the first successful download since we normally
        only need one valid PDF.
        """
        results: list[tuple[SearchCandidate, str]] = []

        for candidate in candidates:
            try:
                logger.info("Trying candidate: %s", candidate.url)
                path = await self._try_download(candidate, progress_callback)
                if path:
                    logger.info("Downloaded %s → %s", candidate.url, path)
                    results.append((candidate, path))
                    break  # one is enough
            except Exception:
                logger.exception("Download failed for %s", candidate.url)

        if not results:
            logger.warning(
                "No PDFs downloaded from %d candidates", len(candidates)
            )
        return results

    async def _try_download(self, candidate: SearchCandidate, progress_callback: Callable[[int, int], None] | None = None) -> str | None:
        """Try direct download first if it looks like a PDF, fallback to Playwright browser download."""
        parsed_url = urlparse(candidate.url.strip())
        is_pdf_path = parsed_url.path.lower().endswith(".pdf")
        
        if candidate.is_direct_pdf or is_pdf_path:
            path = await self._try_direct_download(candidate.url.strip(), progress_callback)
            if path:
                return path
            logger.info("Direct download failed, falling back to browser for: %s", candidate.url)
            
        return await self._try_browser_download(candidate.url.strip())

    # ------------------------------------------------------------------
    # Direct HTTP download
    # ------------------------------------------------------------------

    async def _try_direct_download(
        self, url: str, progress_callback: Callable[[int, int], None] | None = None
    ) -> str | None:
        """Download a direct PDF link with httpx."""
        logger.info("Direct download: %s", url)
        try:
            parsed = urlparse(url)
            filename = os.path.basename(parsed.path)
            if not filename.lower().endswith(".pdf"):
                filename = f"report_{abs(hash(url)) % 10**8}.pdf"

            dest = self.download_dir / filename

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.google.com/"
            }

            async with httpx.AsyncClient(
                verify=False, timeout=60.0, follow_redirects=True, headers=headers
            ) as client:
                async with client.stream("GET", url) as response:
                    response.raise_for_status()
                    total_size = int(response.headers.get("content-length", 0))
                    downloaded = 0
                    
                    with open(dest, "wb") as f:
                        async for chunk in response.aiter_bytes(chunk_size=65536):
                            f.write(chunk)
                            downloaded += len(chunk)
                            if progress_callback:
                                progress_callback(downloaded, total_size)

            logger.info("Saved %d bytes → %s", downloaded, dest)
            return str(dest)

        except httpx.HTTPStatusError as exc:
            logger.error("HTTP %d for %s", exc.response.status_code, url)
        except httpx.RequestError as exc:
            logger.error("Request error downloading %s: %s", url, exc)
        except Exception:
            logger.exception("Unexpected error downloading %s", url)
        return None

    # ------------------------------------------------------------------
    # Playwright browser download
    # ------------------------------------------------------------------

    async def _try_browser_download(self, url: str) -> str | None:
        """Render an HTML page in headless Chromium and grab the PDF."""
        logger.info("Browser download: %s", url)

        try:
            async with async_playwright() as pw:
                browser = await pw.chromium.launch(headless=True)
                context = await browser.new_context(accept_downloads=True)
                page = await context.new_page()
                page.set_default_timeout(30_000)

                # Listen for auto-triggered downloads.
                download_future: asyncio.Future = asyncio.get_event_loop().create_future()

                def _on_download(dl):  # type: ignore[no-untyped-def]
                    if not download_future.done():
                        download_future.set_result(dl)

                page.on("download", _on_download)

                try:
                    await page.goto(url, wait_until="networkidle", timeout=30_000)
                except Exception as nav_err:
                    logger.warning("Navigation issue for %s: %s", url, nav_err)

                download = None

                # 1. Check for auto-download.
                if download_future.done():
                    download = download_future.result()
                else:
                    # 2. Look for PDF links on the page.
                    links = await page.query_selector_all("a")
                    target_link = None

                    for link in links:
                        href = await link.get_attribute("href") or ""
                        text = (await link.inner_text()).lower()
                        href_lower = href.lower()

                        if href_lower.endswith(".pdf"):
                            target_link = link
                            break
                        if (
                            "annual report" in text or "download" in text
                        ) and ".pdf" in href_lower:
                            target_link = link
                            break

                    if target_link:
                        logger.info("Clicking PDF link on page")
                        try:
                            async with page.expect_download(timeout=15_000) as dl_info:
                                await target_link.click()
                            download = await dl_info.value
                        except Exception:
                            logger.warning("Click did not trigger a download")

                if download:
                    fname = download.suggested_filename
                    if not fname.lower().endswith(".pdf"):
                        fname += ".pdf"
                    dest = self.download_dir / fname
                    await download.save_as(str(dest))
                    await browser.close()
                    logger.info("Browser download saved → %s", dest)
                    return str(dest)

                await browser.close()
                logger.info("No PDF found on page %s", url)
                return None

        except Exception:
            logger.exception("Playwright error for %s", url)
        return None
