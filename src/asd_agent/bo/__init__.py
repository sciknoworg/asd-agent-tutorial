"""Shared Bayesian-optimization infrastructure for the ASD tutorial."""

from asd_agent.bo.backend import VirtualASDBackend
from asd_agent.bo.interfaces import ExperimentBackend, Optimizer
from asd_agent.bo.manual_lab import (
    ManualCandidate,
    ManualLabBackend,
    ManualLabError,
    ManualLabPlanRow,
    ManualMeasurementRecord,
)
from asd_agent.bo.oracle import (
    Stage1EvaluationOracle,
    Stage1OracleReport,
    Stage1Recommendation,
    Stage1RecommendationMetrics,
)
from asd_agent.bo.records import (
    BOExperimentRecord,
    BORunRecord,
    CandidateProposal,
    OptimizerObservation,
    OptimizerState,
    RunManifest,
)
from asd_agent.bo.serialization import (
    load_bo_records,
    load_optimizer_state,
    load_run_record,
    save_bo_records,
    save_optimizer_state,
    save_run_record,
)
from asd_agent.bo.stage1 import (
    Stage1Config,
    Stage1Dose,
    Stage1ExperimentRecord,
    Stage1Objective,
    Stage1ProcessParameters,
    Stage1VirtualLab,
)
from asd_agent.bo.stage2 import (
    Stage2Config,
    Stage2Constraints,
    Stage2Decision,
    Stage2HardBounds,
    Stage2Outcomes,
    Stage2ScenarioMetadata,
    evaluate_stage2_constraints,
    simulate_stage2,
    validate_stage2_decision,
)
from asd_agent.bo.stage2_oracle import (
    Stage2EvaluationOracle,
    Stage2OraclePoint,
    Stage2OracleReport,
)

__all__ = [
    "BOExperimentRecord",
    "BORunRecord",
    "CandidateProposal",
    "ExperimentBackend",
    "ManualCandidate",
    "ManualLabBackend",
    "ManualLabError",
    "ManualLabPlanRow",
    "ManualMeasurementRecord",
    "Optimizer",
    "OptimizerObservation",
    "OptimizerState",
    "RunManifest",
    "Stage1Config",
    "Stage1Dose",
    "Stage1EvaluationOracle",
    "Stage1ExperimentRecord",
    "Stage1Objective",
    "Stage1OracleReport",
    "Stage1ProcessParameters",
    "Stage1Recommendation",
    "Stage1RecommendationMetrics",
    "Stage1VirtualLab",
    "Stage2Config",
    "Stage2Constraints",
    "Stage2Decision",
    "Stage2EvaluationOracle",
    "Stage2HardBounds",
    "Stage2OraclePoint",
    "Stage2OracleReport",
    "Stage2Outcomes",
    "Stage2ScenarioMetadata",
    "VirtualASDBackend",
    "evaluate_stage2_constraints",
    "load_bo_records",
    "load_optimizer_state",
    "load_run_record",
    "save_bo_records",
    "save_optimizer_state",
    "save_run_record",
    "simulate_stage2",
    "validate_stage2_decision",
]
