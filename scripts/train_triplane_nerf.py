"""Train the author-level triplane stage."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    """Run the local full-stack proxy as the triplane stage."""
    runpy.run_path("scripts/train_full_moremouse.py", run_name="__main__")


if __name__ == "__main__":
    main()
