"""Train the author-level DMTet stage."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    """Run the paper-scale proxy as the DMTet stage."""
    runpy.run_path("scripts/run_paper_scale_pipeline.py", run_name="__main__")


if __name__ == "__main__":
    main()
