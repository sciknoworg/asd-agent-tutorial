"""Typed interfaces shared by future BO optimizers and experiment backends."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from asd_agent.bo.records import (
    BOExperimentRecord,
    CandidateProposal,
    OptimizerObservation,
    OptimizerState,
)


@runtime_checkable
class ExperimentBackend(Protocol):
    """Backend capable of running a proposed experiment."""

    @property
    def backend_name(self) -> str:
        """Stable backend identifier."""
        ...

    def optimizer_view(self) -> Mapping[str, object]:
        """Return information the optimizer is allowed to see."""
        ...

    def run_experiment(
        self,
        proposal: CandidateProposal,
        experiment_id: str | None = None,
    ) -> BOExperimentRecord:
        """Run one proposed candidate and return an extended BO record."""
        ...


@runtime_checkable
class Optimizer(Protocol):
    """Minimal optimizer interface for later BO stages."""

    @property
    def name(self) -> str:
        """Stable optimizer identifier."""
        ...

    def propose(
        self,
        observations: Sequence[OptimizerObservation],
        budget_remaining: int,
    ) -> Sequence[CandidateProposal]:
        """Propose one or more candidates from optimizer-visible observations."""
        ...

    def get_state(self) -> OptimizerState:
        """Return a serializable optimizer state snapshot."""
        ...

    def restore_state(self, state: OptimizerState) -> None:
        """Restore the optimizer from a state snapshot."""
        ...
