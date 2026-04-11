#!/usr/bin/env python3
"""Test crawl4ai on the distribution-overview page.

Compares different crawl configs to find why "When do I get paid?" section
is missing from stored chunks. Also tests with Amazon Nova 2 Lite as the
LLM extraction strategy.

Usage:
    cd apps/rag-service
    uv run python3 scripts/test_crawl_distribution.py
"""

from __future__ import annotations

import asyncio
import os
import sys

TARGET_URL = (
    "https://www.apraamcos.com.au/music-creators/membership-explained/distribution-overview"
)

# Phrases we expect to find on the page (from Playwright snapshot)
EXPECTED_PHRASES = [
    "When do I get paid",
    "every 3 months",
    "Domestic royalties",
    "International royalties",
    "each month",
    "13.26%",
    "86 cents",
    "Expense to revenue ratio",
    "128,000+",
]


def check_content(label: str, text: str) -> dict[str, bool]:
    """Check which expected phrases appear in extracted text."""
    results: dict[str, bool] = {}
    text_lower = text.lower()
    for phrase in EXPECTED_PHRASES:
        results[phrase] = phrase.lower() in text_lower
    found = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\n{'='*60}")
    print(f"[{label}] {found}/{total} expected phrases found  ({len(text)} chars)")
    print(f"{'='*60}")
    for phrase, ok in results.items():
        status = "✅" if ok else "❌"
        print(f"  {status} {phrase}")
    return results


async def run_crawl4ai_basic() -> str:
    """Config 1: Reproduce existing crawler settings (text_mode, excluded_tags)."""
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig

    browser_config = BrowserConfig(
        headless=True,
        text_mode=True,
        light_mode=True,
        extra_args=["--disable-extensions", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(
        excluded_tags=["script", "style", "nav", "footer", "header"],
        exclude_external_links=True,
        exclude_all_images=True,
        word_count_threshold=50,
        page_timeout=30000,
        delay_before_return_html=0.5,
        cache_mode=CacheMode.BYPASS,
    )
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=TARGET_URL, config=crawl_config)
        return result.markdown or result.extracted_content or ""


async def run_crawl4ai_full_render() -> str:
    """Config 2: Full JS render, no excluded tags, longer wait."""
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig

    browser_config = BrowserConfig(
        headless=True,
        text_mode=False,  # full render, not text-only
        light_mode=False,
        extra_args=["--disable-extensions", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(
        excluded_tags=["script", "style"],  # only exclude scripts/styles
        exclude_external_links=True,
        exclude_all_images=True,
        word_count_threshold=10,  # lower threshold
        page_timeout=60000,  # longer timeout
        delay_before_return_html=3.0,  # wait 3s for JS to render
        cache_mode=CacheMode.BYPASS,
    )
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=TARGET_URL, config=crawl_config)
        return result.markdown or result.extracted_content or ""


async def run_crawl4ai_with_js_wait() -> str:
    """Config 3: Execute JS to scroll page and wait for dynamic content."""
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig

    browser_config = BrowserConfig(
        headless=True,
        text_mode=False,
        light_mode=False,
        extra_args=["--disable-extensions", "--no-sandbox"],
    )

    # JS that scrolls to bottom to trigger lazy-loaded content
    scroll_js = """
    (async () => {
        const delay = (ms) => new Promise(r => setTimeout(r, ms));
        for (let i = 0; i < 5; i++) {
            window.scrollBy(0, window.innerHeight);
            await delay(500);
        }
        window.scrollTo(0, 0);
        await delay(500);
    })();
    """

    crawl_config = CrawlerRunConfig(
        excluded_tags=["script", "style"],
        exclude_external_links=True,
        exclude_all_images=True,
        word_count_threshold=10,
        page_timeout=60000,
        delay_before_return_html=5.0,  # wait 5s after scroll
        js_code=[scroll_js],
        cache_mode=CacheMode.BYPASS,
    )
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=TARGET_URL, config=crawl_config)
        return result.markdown or result.extracted_content or ""


async def run_crawl4ai_llm_extraction() -> str:
    """Config 4: Use Amazon Nova 2 Lite via Bedrock for LLM-based extraction."""
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
    from crawl4ai.extraction_strategy import LLMExtractionStrategy

    browser_config = BrowserConfig(
        headless=True,
        text_mode=False,
        light_mode=False,
        extra_args=["--disable-extensions", "--no-sandbox"],
    )

    # Use Amazon Nova 2 Lite via litellm bedrock provider
    llm_strategy = LLMExtractionStrategy(
        provider="bedrock/amazon.nova-2-lite-v1:0",
        instruction=(
            "Extract ALL text content from this webpage about APRA AMCOS royalty distribution. "
            "Include every section, especially: "
            "1) Expense to revenue ratio "
            "2) How we collect data "
            "3) APRA AMCOS distribution process "
            "4) Who and what APRA AMCOS collects and distributes royalties for "
            "5) When do I get paid (domestic and international royalties schedule) "
            "6) Any payment frequency information (quarterly, monthly, etc.) "
            "Return the full text content preserving section headings."
        ),
    )

    scroll_js = """
    (async () => {
        const delay = (ms) => new Promise(r => setTimeout(r, ms));
        for (let i = 0; i < 5; i++) {
            window.scrollBy(0, window.innerHeight);
            await delay(500);
        }
        window.scrollTo(0, 0);
        await delay(500);
    })();
    """

    crawl_config = CrawlerRunConfig(
        excluded_tags=["script", "style"],
        exclude_external_links=True,
        exclude_all_images=True,
        word_count_threshold=10,
        page_timeout=60000,
        delay_before_return_html=5.0,
        js_code=[scroll_js],
        extraction_strategy=llm_strategy,
        cache_mode=CacheMode.BYPASS,
    )
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=TARGET_URL, config=crawl_config)
        # LLM extraction goes to extracted_content; markdown is the raw
        return result.extracted_content or result.markdown or ""


async def main() -> None:
    print(f"Target URL: {TARGET_URL}")
    print(f"Expected phrases: {len(EXPECTED_PHRASES)}")

    # Ensure AWS region is set for Bedrock calls
    os.environ.setdefault("AWS_DEFAULT_REGION", "ap-southeast-2")
    os.environ.setdefault("AWS_REGION", "ap-southeast-2")

    configs = [
        ("1-BASIC (current crawler settings)", run_crawl4ai_basic),
        ("2-FULL_RENDER (no text_mode, fewer excludes)", run_crawl4ai_full_render),
        ("3-JS_SCROLL (scroll + 5s wait)", run_crawl4ai_with_js_wait),
    ]

    # Only add LLM config if user passes --llm flag
    use_llm = "--llm" in sys.argv
    if use_llm:
        configs.append(("4-LLM_NOVA2LITE (Bedrock extraction)", run_crawl4ai_llm_extraction))
    else:
        print("\nTip: pass --llm to also test Amazon Nova 2 Lite LLM extraction")

    all_results: dict[str, dict[str, bool]] = {}
    all_texts: dict[str, str] = {}

    for label, fn in configs:
        print(f"\n{'#'*60}")
        print(f"# Running: {label}")
        print(f"{'#'*60}")
        try:
            text = await fn()
            all_texts[label] = text
            all_results[label] = check_content(label, text)
        except Exception as e:
            print(f"\n❌ FAILED: {label}")
            print(f"   Error: {e}")
            all_results[label] = {p: False for p in EXPECTED_PHRASES}

    # Summary table
    print(f"\n\n{'='*70}")
    print("SUMMARY — Expected Phrase Coverage")
    print(f"{'='*70}")
    header = f"{'Phrase':<30}"
    for label in all_results:
        short = label.split("(")[0].strip()
        header += f" | {short:>12}"
    print(header)
    print("-" * len(header))
    for phrase in EXPECTED_PHRASES:
        row = f"{phrase:<30}"
        for label in all_results:
            ok = all_results[label].get(phrase, False)
            row += f" | {'✅':>12}" if ok else f" | {'❌':>12}"
        print(row)

    # Score summary
    print()
    for label in all_results:
        found = sum(1 for v in all_results[label].values() if v)
        total = len(EXPECTED_PHRASES)
        short = label.split("(")[0].strip()
        bar = "█" * found + "░" * (total - found)
        print(f"  {short:>20}: {bar} {found}/{total}")

    # Save raw text to tmp for inspection
    out_dir = "tmp/crawl_test"
    os.makedirs(out_dir, exist_ok=True)
    for label, text in all_texts.items():
        safe_name = label.split("(")[0].strip().replace(" ", "_").lower()
        path = os.path.join(out_dir, f"{safe_name}.txt")
        with open(path, "w") as f:
            f.write(text)
        print(f"\n  Saved raw text: {path} ({len(text)} chars)")


if __name__ == "__main__":
    asyncio.run(main())
