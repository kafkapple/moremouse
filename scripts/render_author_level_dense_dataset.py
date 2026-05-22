"""Render the author-level dense supervision dataset."""

from __future__ import annotations

import runpy


def main() -> None:
    """Run the paper-scale dense rendering pipeline as the dense-dataset stage."""
    runpy.run_path("scripts/run_paper_scale_pipeline.py", run_name="__main__")


if __name__ == "__main__":
    main()
