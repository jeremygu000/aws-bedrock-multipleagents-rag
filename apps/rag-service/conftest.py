"""Root conftest for rag-service tests.

Ensures that standalone modules like ``ingestion_handler`` (which live at the
project root but outside the ``app`` package) are importable during test
collection.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add the rag-service project root to sys.path so that top-level modules
# (e.g. ingestion_handler, lambda_tool) can be imported by tests.
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
