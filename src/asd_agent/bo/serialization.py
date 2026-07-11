"""JSON serialization helpers for BO records and optimizer state."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from asd_agent.bo.records import BOExperimentRecord, BORunRecord, OptimizerState


def save_model(model: BaseModel, path: str | Path) -> None:
    """Save one Pydantic model as indented JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(model.model_dump_json(indent=2), encoding="utf-8")


def save_optimizer_state(state: OptimizerState, path: str | Path) -> None:
    """Save an optimizer checkpoint."""

    save_model(state, path)


def load_optimizer_state(path: str | Path) -> OptimizerState:
    """Load an optimizer checkpoint."""

    source = Path(path)
    return OptimizerState.model_validate_json(source.read_text(encoding="utf-8"))


def save_bo_records(records: Sequence[BOExperimentRecord], path: str | Path) -> None:
    """Save BO experiment records as a JSON list."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = [record.model_dump(mode="json") for record in records]
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_bo_records(path: str | Path) -> list[BOExperimentRecord]:
    """Load BO experiment records from a JSON list."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("BO record file must contain a JSON list")
    return [BOExperimentRecord.model_validate(item) for item in payload]


def save_run_record(record: BORunRecord, path: str | Path) -> None:
    """Save a BO run record."""

    save_model(record, path)


def load_run_record(path: str | Path) -> BORunRecord:
    """Load a BO run record."""

    source = Path(path)
    return BORunRecord.model_validate_json(source.read_text(encoding="utf-8"))
