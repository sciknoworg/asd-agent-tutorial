"""Experiment backends for BO-capable workflows."""

from __future__ import annotations

from asd_agent.bo.records import BOExperimentRecord, CandidateProposal
from asd_agent.models import ProcessConfig
from asd_agent.objective import validate_safety
from asd_agent.simulator import VirtualLab


class VirtualASDBackend:
    """BO backend that delegates all measurements to the existing virtual lab."""

    def __init__(
        self,
        config: ProcessConfig,
        *,
        seed: int | None = None,
        backend_name: str = "virtual_asd",
    ) -> None:
        self.config = config
        self.seed = config.seed if seed is None else seed
        self._backend_name = backend_name
        self._lab = VirtualLab(config, seed=self.seed)

    @property
    def backend_name(self) -> str:
        """Stable backend identifier."""

        return self._backend_name

    def optimizer_view(self) -> dict[str, object]:
        """Return optimizer-facing context without hidden simulator parameters."""

        return {
            "backend": self.backend_name,
            "scenario": self.config.scenario,
            "description": self.config.description,
            "objective": self.config.objective.model_dump(mode="json"),
            "safety_bounds": self.config.safety.model_dump(mode="json"),
        }

    def hidden_simulator_parameters(self) -> dict[str, object]:
        """Return simulator-only parameters that must not be passed to optimizers."""

        return {
            "surfaces": {
                name: params.model_dump(mode="json")
                for name, params in self.config.surfaces.items()
            },
            "noise_sigma_nm": self.config.noise_sigma_nm,
            "per_cycle_overhead_s": self.config.per_cycle_overhead_s,
            "stabilization_time_s": self.config.stabilization_time_s,
        }

    def run_experiment(
        self,
        proposal: CandidateProposal,
        experiment_id: str | None = None,
    ) -> BOExperimentRecord:
        """Validate and run a proposal through the existing simulator."""

        violations = validate_safety(proposal.parameters, self.config)
        if violations:
            raise ValueError(f"unsafe BO candidate: {violations}")
        record = self._lab.simulate(
            proposal.parameters,
            experiment_id=experiment_id or proposal.candidate_id,
            decision_rationale=f"BO proposal from {proposal.optimizer}.",
        )
        return BOExperimentRecord.from_experiment_record(
            record,
            proposal=proposal,
            metadata={"backend": self.backend_name},
        )
