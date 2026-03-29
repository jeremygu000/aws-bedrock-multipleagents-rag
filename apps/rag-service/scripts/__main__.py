"""Allow running as `python -m scripts.crawl_ingest`."""

from __future__ import annotations

import asyncio
import sys

from .crawl_ingest import main

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
