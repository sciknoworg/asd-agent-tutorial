"""Scenario configuration loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from asd_agent.models import ProcessConfig

if TYPE_CHECKING:
    from asd_agent.bo.stage1 import Stage1Config
    from asd_agent.bo.stage2 import Stage2Config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_config(path: str | Path) -> ProcessConfig:
    """Load a scenario config from JSON-compatible YAML."""

    return ProcessConfig.model_validate(load_config_mapping(path))


def load_config_mapping(path: str | Path) -> dict[str, Any]:
    """Load a JSON-compatible YAML file as a mapping."""

    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    loaded: Any
    try:
        import yaml  # type: ignore[import-untyped]

        loaded = yaml.safe_load(text)
    except ModuleNotFoundError:
        loaded = json.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError("scenario config must be a mapping")
    return loaded


def load_scenario(name: str) -> ProcessConfig:
    """Load one of the predefined tutorial scenarios."""

    path = CONFIG_DIR / f"{name}.yaml"
    if name.startswith("bo_stage1_"):
        raise FileNotFoundError(f"{name!r} is a Stage 1 scenario; use load_stage1_scenario instead")
    if name.startswith("bo_stage2_"):
        raise FileNotFoundError(f"{name!r} is a Stage 2 scenario; use load_stage2_scenario instead")
    if not path.exists():
        available = sorted(
            item.stem
            for item in CONFIG_DIR.glob("*.yaml")
            if not item.stem.startswith(("bo_stage1_", "bo_stage2_"))
        )
        raise FileNotFoundError(f"unknown scenario {name!r}; available: {available}")
    return load_config(path)


def load_stage1_config(path: str | Path) -> Stage1Config:
    """Load a Stage 1 saturation-process config from JSON-compatible YAML."""

    from asd_agent.bo.stage1 import Stage1Config

    return Stage1Config.model_validate(load_config_mapping(path))


def load_stage1_scenario(name: str) -> Stage1Config:
    """Load one of the BO Stage 1 scenarios."""

    stem = name if name.startswith("bo_stage1_") else f"bo_stage1_{name}"
    path = CONFIG_DIR / f"{stem}.yaml"
    if not path.exists():
        available = sorted(
            item.stem.removeprefix("bo_stage1_") for item in CONFIG_DIR.glob("bo_stage1_*.yaml")
        )
        raise FileNotFoundError(f"unknown Stage 1 scenario {name!r}; available: {available}")
    return load_stage1_config(path)


def load_stage2_config(path: str | Path) -> Stage2Config:
    """Load a Stage 2 constrained ASD config from JSON-compatible YAML."""

    from asd_agent.bo.stage2 import Stage2Config

    return Stage2Config.model_validate(load_config_mapping(path))


def load_stage2_scenario(name: str) -> Stage2Config:
    """Load one of the BO Stage 2 scenarios."""

    stem = name if name.startswith("bo_stage2_") else f"bo_stage2_{name}"
    path = CONFIG_DIR / f"{stem}.yaml"
    if not path.exists():
        available = sorted(
            item.stem.removeprefix("bo_stage2_") for item in CONFIG_DIR.glob("bo_stage2_*.yaml")
        )
        raise FileNotFoundError(f"unknown Stage 2 scenario {name!r}; available: {available}")
    return load_stage2_config(path)
