"""Generic Gaussian-process helpers for Stage 1 active learning."""

from __future__ import annotations

import warnings as warning_capture
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import gpytorch
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from pydantic import BaseModel, ConfigDict, Field

from asd_agent.bo.stage1 import Stage1ExperimentRecord
from asd_agent.models import Range

GPNoiseMode = Literal["known", "inferred"]


class GenericGPSettings(BaseModel):
    """Settings for a one-dimensional generic GP surrogate."""

    noise_mode: GPNoiseMode = "inferred"
    known_noise_variance: float | None = Field(default=None, ge=0.0)
    min_known_noise_variance: float = Field(default=1e-9, gt=0.0)
    max_fit_attempts: int = Field(default=2, ge=1, le=5)

    model_config = ConfigDict(extra="forbid")


class GPPrediction(BaseModel):
    """Finite posterior summaries on a dose grid."""

    dose_s: list[float]
    mean: list[float]
    stddev: list[float]

    model_config = ConfigDict(extra="forbid")


class GPFitFailure(RuntimeError):
    """Raised when all GP fitting attempts fail."""


@dataclass(frozen=True)
class GenericGPModel:
    """A fitted Stage 1 GP and its reproducibility metadata."""

    model: Any
    bounds: Range
    settings: GenericGPSettings
    fit_warnings: list[str]
    training_observation_ids: list[str]

    def posterior(self, doses_s: Sequence[float]) -> GPPrediction:
        """Evaluate finite posterior mean and standard deviation."""

        if not doses_s:
            raise ValueError("posterior requires at least one dose")
        query_x = dose_tensor(doses_s)
        self.model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = self.model.posterior(query_x)
        mean = posterior.mean.squeeze(-1).detach().cpu()
        variance = posterior.variance.clamp_min(0.0).squeeze(-1).detach().cpu()
        stddev = torch.sqrt(variance)
        return GPPrediction(
            dose_s=[float(value) for value in doses_s],
            mean=[float(value) for value in mean.tolist()],
            stddev=[float(value) for value in stddev.tolist()],
        )


def fit_generic_stage1_gp(
    records: Sequence[Stage1ExperimentRecord],
    bounds: Range,
    settings: GenericGPSettings | None = None,
    *,
    seed: int | None = None,
) -> GenericGPModel:
    """Fit a float64 one-dimensional Matern GP to Stage 1 observations."""

    resolved_settings = settings or GenericGPSettings()
    train_x, train_y = training_tensors(records, bounds)
    train_yvar = known_noise_tensor(train_y, resolved_settings)
    fit_warnings: list[str] = []
    last_error: Exception | None = None

    for attempt in range(resolved_settings.max_fit_attempts):
        if seed is not None:
            torch.manual_seed(seed + attempt)
        model = build_generic_model(train_x, train_y, train_yvar, bounds)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        try:
            with warning_capture.catch_warnings(record=True) as caught:
                warning_capture.simplefilter("always")
                fit_gpytorch_mll(mll)
            fit_warnings.extend(
                f"attempt {attempt + 1}: {item.category.__name__}: {item.message}"
                for item in caught
            )
            model.eval()
            return GenericGPModel(
                model=model,
                bounds=bounds,
                settings=resolved_settings,
                fit_warnings=fit_warnings,
                training_observation_ids=[record.experiment_id for record in records],
            )
        except Exception as exc:  # pragma: no cover - exercised with monkeypatch tests
            last_error = exc
            fit_warnings.append(f"attempt {attempt + 1} failed: {type(exc).__name__}: {exc}")

    message = "generic GP fitting failed"
    if last_error is not None:
        message = f"{message}: {last_error}"
    raise GPFitFailure(message)


def build_generic_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    train_yvar: torch.Tensor | None,
    bounds: Range,
) -> SingleTaskGP:
    """Build a normalized-input, standardized-output Matern GP."""

    bound_tensor = torch.tensor([[bounds.min], [bounds.max]], dtype=torch.double)
    covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=1))
    return SingleTaskGP(
        train_X=train_x,
        train_Y=train_y,
        train_Yvar=train_yvar,
        covar_module=covar_module,
        input_transform=Normalize(d=1, bounds=bound_tensor),
        outcome_transform=Standardize(m=1),
    )


def training_tensors(
    records: Sequence[Stage1ExperimentRecord],
    bounds: Range,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert optimizer-facing Stage 1 records into float64 tensors."""

    if len(records) < 2:
        raise ValueError("generic GP fitting requires at least two observations")
    for record in records:
        if not bounds.contains(record.dose_s):
            raise ValueError(f"dose outside GP bounds: {record.dose_s}")
    train_x = dose_tensor([record.dose_s for record in records])
    train_y = torch.tensor(
        [[record.observed_growth] for record in records],
        dtype=torch.double,
    )
    return train_x, train_y


def dose_tensor(doses_s: Sequence[float]) -> torch.Tensor:
    """Return a column vector of float64 doses."""

    return torch.tensor([[float(dose)] for dose in doses_s], dtype=torch.double)


def known_noise_tensor(
    train_y: torch.Tensor,
    settings: GenericGPSettings,
) -> torch.Tensor | None:
    """Return a known-noise tensor, or None for inferred-noise mode."""

    if settings.noise_mode == "inferred":
        return None
    if settings.known_noise_variance is None:
        raise ValueError("known-noise GP mode requires known_noise_variance")
    variance = max(settings.known_noise_variance, settings.min_known_noise_variance)
    return torch.full_like(train_y, variance, dtype=torch.double)
