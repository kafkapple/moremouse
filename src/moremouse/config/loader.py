"""OmegaConf-based configuration loading."""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_config(config_path: Path) -> DictConfig:
    """Load a readonly OmegaConf config.

    Parameters
    ----------
    config_path:
        YAML configuration path.

    Returns
    -------
    DictConfig
        Readonly resolved configuration.

    Raises
    ------
    FileNotFoundError
        If the config path does not exist.
    ValueError
        If required top-level sections are missing.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    cfg = OmegaConf.load(config_path)
    missing = {"seed", "paths", "dataset", "render", "model", "training"} - set(cfg.keys())
    if missing:
        raise ValueError(f"Config missing required sections: {sorted(missing)}")
    OmegaConf.set_readonly(cfg, True)
    return cfg

