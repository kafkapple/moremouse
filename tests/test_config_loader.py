from pathlib import Path

from omegaconf import OmegaConf

from moremouse.config import load_config


def test_load_config_readonly() -> None:
    """Load the default config as a readonly object."""
    cfg = load_config(Path("configs/default.yaml"))

    assert cfg.seed == int("260520")
    assert OmegaConf.is_readonly(cfg)
