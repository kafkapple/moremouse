"""
Environment-aware Configuration Loader

Automatically detects the environment (gpu05, local, etc.) based on hostname
and loads the appropriate configuration file.

Usage:
    from src.utils.config import load_config, get_paths

    # Load full config
    config = load_config()

    # Get paths for current environment
    paths = get_paths()
    data_dir = paths['data_dir']
"""

import os
import socket
from pathlib import Path
from typing import Dict, Optional, Any

import yaml


def get_hostname() -> str:
    """Get the current machine hostname."""
    return socket.gethostname()


def detect_environment() -> str:
    """
    Detect the current environment based on hostname.

    Returns:
        Environment name: 'gpu05', 'local', or 'default'
    """
    hostname = get_hostname().lower()

    # Check known patterns
    if 'gpu05' in hostname or 'dlbox' in hostname:
        return 'gpu05'
    elif 'joon-dell' in hostname or 'dell' in hostname:
        return 'local'

    # Fallback to default
    return 'default'


def get_config_dir() -> Path:
    """Get the configs directory path."""
    # configs/ is at project root
    src_dir = Path(__file__).parent.parent  # src/utils -> src
    project_root = src_dir.parent  # src -> project root
    return project_root / 'configs'


def load_yaml(path: Path) -> Dict:
    """Load a YAML file."""
    if not path.exists():
        return {}
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Deep merge two dictionaries.
    override values take precedence over base values.
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def expand_paths(paths: Dict) -> Dict:
    """Expand ~ and environment variables in paths."""
    expanded = {}
    for key, value in paths.items():
        if isinstance(value, str):
            expanded[key] = os.path.expanduser(os.path.expandvars(value))
        else:
            expanded[key] = value
    return expanded


def load_config(environment: Optional[str] = None) -> Dict:
    """
    Load configuration for the specified or detected environment.

    Args:
        environment: Optional environment name. If None, auto-detect.

    Returns:
        Merged configuration dictionary
    """
    config_dir = get_config_dir()

    # Auto-detect environment if not specified
    if environment is None:
        environment = detect_environment()

    # Load default config first
    default_config = load_yaml(config_dir / 'default.yaml')

    # Load environment-specific config
    env_config = load_yaml(config_dir / f'{environment}.yaml')

    # Merge configs (env overrides default)
    config = deep_merge(default_config, env_config)

    # Expand paths
    if 'paths' in config:
        config['paths'] = expand_paths(config['paths'])

    # Add detected environment to config
    config['_environment'] = environment
    config['_hostname'] = get_hostname()

    return config


def get_paths(environment: Optional[str] = None) -> Dict[str, str]:
    """
    Get paths for the current environment.

    Returns:
        Dictionary of path names to expanded path strings
    """
    config = load_config(environment)
    return config.get('paths', {})


def get_device(environment: Optional[str] = None) -> str:
    """Get the device setting for current environment."""
    config = load_config(environment)
    return config.get('device', 'cuda:0')


def print_config(environment: Optional[str] = None):
    """Print current configuration for debugging."""
    config = load_config(environment)

    print(f"\n{'='*60}")
    print(f"MoReMouse Configuration")
    print(f"{'='*60}")
    print(f"Environment: {config.get('_environment', 'unknown')}")
    print(f"Hostname: {config.get('_hostname', 'unknown')}")
    print(f"\nPaths:")
    for key, value in config.get('paths', {}).items():
        print(f"  {key}: {value}")
    print(f"\nDevice: {config.get('device', 'cuda:0')}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    # Test config loading
    print_config()
