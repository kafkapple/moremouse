"""Run the staged author-level MoReMouse reproduction pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys

from loguru import logger
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class Stage:
    """One reproducibility stage."""

    name: str
    command: list[str]


def main() -> None:
    """Run the staged pipeline with resource checks and a stage report."""
    cfg = OmegaConf.load("configs/default.yaml")
    stages = build_stages()
    output_root = Path(cfg.paths.gpu_result_root) / "experiments" / "full_reproduction_260522"
    output_root.mkdir(parents=True, exist_ok=True)
    stage_report = {"stages": [], "resource_snapshots": []}
    for stage in stages:
        snapshot = resource_snapshot()
        stage_report["resource_snapshots"].append(snapshot)
        logger.info("preflight {} {}", stage.name, snapshot)
        run_stage(stage)
        stage_report["stages"].append({"name": stage.name, "command": stage.command})
    (output_root / "report.json").write_text(json.dumps(stage_report, indent=2), encoding="utf-8")
    logger.info("Wrote reproduction stage report to {}", output_root / "report.json")


def build_stages() -> list[Stage]:
    """Return the ordered stage list."""
    python = sys.executable
    return [
        Stage("agam", [python, "scripts/train_agam.py"]),
        Stage("dense_dataset", [python, "scripts/render_author_level_dense_dataset.py"]),
        Stage("triplane", [python, "scripts/train_triplane_nerf.py"]),
        Stage("dmtet", [python, "scripts/train_dmtet.py"]),
    ]


def run_stage(stage: Stage) -> None:
    """Execute one stage and fail fast on errors."""
    logger.info("starting stage {}", stage.name)
    subprocess.run(stage.command, check=True)


def resource_snapshot() -> dict[str, object]:
    """Collect a compact gpu03 resource snapshot."""
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.free,memory.total",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip().splitlines()
    disk = subprocess.check_output(["df", "-h", "/home/joon", "/home/joon/results"], text=True).strip().splitlines()
    return {"gpu": gpu, "disk": disk}


if __name__ == "__main__":
    main()
