"""Scenario configuration loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from asd_agent.models import ProcessConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_config(path: str | Path) -> ProcessConfig:
    """Load a scenario config from JSON-compatible YAML."""

    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    data: dict[str, Any]
    try:
        import yaml  # type: ignore[import-untyped]

        loaded = yaml.safe_load(text)
        if not isinstance(loaded, dict):
            raise ValueError("scenario config must be a mapping")
        data = loaded
    except ModuleNotFoundError:
        data = json.loads(text)
    return ProcessConfig.model_validate(data)


def load_scenario(name: str) -> ProcessConfig:
    """Load one of the predefined tutorial scenarios."""

    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        available = sorted(item.stem for item in CONFIG_DIR.glob("*.yaml"))
        raise FileNotFoundError(f"unknown scenario {name!r}; available: {available}")
    return load_config(path)
