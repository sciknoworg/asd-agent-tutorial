"""Physics-informed Stage 1 saturation Gaussian process models."""

from __future__ import annotations

import warnings as warning_capture
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any, cast

import gpytorch
import torch
from gpytorch.constraints import Positive
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from pydantic import BaseModel, ConfigDict, Field

from asd_agent.bo.gp import GPNoiseMode, GPPrediction, dose_tensor
from asd_agent.bo.stage1 import Stage1ExperimentRecord
from asd_agent.models import Range


class PhysicsInformedGPSettings(BaseModel):
    """Settings for the Stage 1 physics-informed GP surrogate."""

    noise_mode: GPNoiseMode = "inferred"
    known_noise_variance: float | None = Field(default=None, ge=0.0)
    min_known_noise_variance: float = Field(default=1e-9, gt=0.0)
    max_fit_attempts: int = Field(default=2, ge=1, le=5)
    max_training_steps: int = Field(default=75, ge=1, le=500)
    learning_rate: float = Field(default=0.05, gt=0.0)
    min_initial_g_inf: float = Field(default=1e-4, gt=0.0)
    min_initial_k: float = Field(default=1e-4, gt=0.0)
    fallback_to_generic: bool = True

    model_config = ConfigDict(extra="forbid")


class PhysicsGPFitFailure(RuntimeError):
    """Raised when all physics-informed GP fitting attempts fail."""


class SaturatingMean(gpytorch.means.Mean):  # type: ignore[misc]
    """Trainable saturating mean: m(t) = g_inf * (1 - exp(-k * t))."""

    def __init__(self, initial_g_inf: float, initial_k: float) -> None:
        super().__init__()
        positive = Positive()
        self.register_parameter(
            "raw_g_inf",
            torch.nn.Parameter(positive.inverse_transform(torch.as_tensor(initial_g_inf))),
        )
        self.register_parameter(
            "raw_k",
            torch.nn.Parameter(positive.inverse_transform(torch.as_tensor(initial_k))),
        )
        self.register_constraint("raw_g_inf", positive)
        self.register_constraint("raw_k", Positive())

    @property
    def g_inf(self) -> torch.Tensor:
        """Positive saturation scale."""

        return cast(torch.Tensor, self.raw_g_inf_constraint.transform(self.raw_g_inf))

    @property
    def k(self) -> torch.Tensor:
        """Positive saturation rate."""

        return cast(torch.Tensor, self.raw_k_constraint.transform(self.raw_k))

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        dose = input_tensor.squeeze(-1).clamp_min(0.0)
        return self.g_inf * (1.0 - torch.exp(-self.k * dose))


class SaturationResidualGP(ExactGP):  # type: ignore[misc]
    """Exact GP with a saturating physical mean and stationary residual kernel."""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: GaussianLikelihood | FixedNoiseGaussianLikelihood,
        mean_module: SaturatingMean,
    ) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module
        self.covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=1))

    def forward(self, input_tensor: torch.Tensor) -> MultivariateNormal:
        mean = self.mean_module(input_tensor)
        covariance = self.covar_module(input_tensor)
        return MultivariateNormal(mean, covariance)


@dataclass(frozen=True)
class PhysicsInformedGPModel:
    """Fitted physics-informed GP plus fit metadata."""

    model: Any
    likelihood: Any
    bounds: Range
    settings: PhysicsInformedGPSettings
    fit_warnings: list[str]
    training_observation_ids: list[str]

    def posterior(self, doses_s: Sequence[float]) -> GPPrediction:
        """Evaluate finite posterior mean and standard deviation."""

        if not doses_s:
            raise ValueError("posterior requires at least one dose")
        query_x = dose_tensor(doses_s)
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            distribution = self.likelihood(self.model(query_x))
        mean = distribution.mean.detach().cpu()
        variance = distribution.variance.clamp_min(0.0).detach().cpu()
        stddev = torch.sqrt(variance)
        return GPPrediction(
            dose_s=[float(value) for value in doses_s],
            mean=[float(value) for value in mean.tolist()],
            stddev=[float(value) for value in stddev.tolist()],
        )

    def physical_parameters(self) -> dict[str, float]:
        """Return fitted physical mean parameters."""

        mean_module = self.model.mean_module
        return {
            "g_inf": float(mean_module.g_inf.detach().cpu()),
            "k": float(mean_module.k.detach().cpu()),
        }


def fit_physics_informed_stage1_gp(
    records: Sequence[Stage1ExperimentRecord],
    bounds: Range,
    settings: PhysicsInformedGPSettings | None = None,
    *,
    seed: int | None = None,
) -> PhysicsInformedGPModel:
    """Fit a physics-informed GP with a trainable saturating mean."""

    resolved_settings = settings or PhysicsInformedGPSettings()
    train_x, train_y = physics_training_tensors(records, bounds)
    train_yvar: torch.Tensor | None = None
    if resolved_settings.noise_mode == "known":
        if resolved_settings.known_noise_variance is None:
            raise ValueError("known-noise physics GP mode requires known_noise_variance")
        variance = max(
            resolved_settings.known_noise_variance,
            resolved_settings.min_known_noise_variance,
        )
        train_yvar = torch.full_like(train_y, variance, dtype=torch.double)
    fit_warnings: list[str] = []
    last_error: Exception | None = None

    for attempt in range(resolved_settings.max_fit_attempts):
        if seed is not None:
            torch.manual_seed(seed + attempt)
        initial_g_inf, initial_k = initialize_physical_parameters(
            records, bounds, resolved_settings
        )
        jitter = 1.0 + 0.10 * attempt
        likelihood: GaussianLikelihood | FixedNoiseGaussianLikelihood
        if train_yvar is None:
            likelihood = GaussianLikelihood()
        else:
            likelihood = FixedNoiseGaussianLikelihood(
                noise=train_yvar.clamp_min(resolved_settings.min_known_noise_variance),
                learn_additional_noise=False,
            )
        mean_module = SaturatingMean(initial_g_inf * jitter, initial_k / jitter).double()
        model = SaturationResidualGP(train_x, train_y, likelihood, mean_module).double()
        mll = ExactMarginalLogLikelihood(likelihood, model)
        try:
            model.train()
            likelihood.train()
            with warning_capture.catch_warnings(record=True) as caught:
                warning_capture.simplefilter("always")
                train_exact_gp(mll, model, train_x, train_y, resolved_settings)
            fit_warnings.extend(
                f"attempt {attempt + 1}: {item.category.__name__}: {item.message}"
                for item in caught
            )
            fitted = PhysicsInformedGPModel(
                model=model.eval(),
                likelihood=likelihood.eval(),
                bounds=bounds,
                settings=resolved_settings,
                fit_warnings=fit_warnings,
                training_observation_ids=[record.experiment_id for record in records],
            )
            parameters = fitted.physical_parameters()
            if not all(isfinite(value) and value > 0.0 for value in parameters.values()):
                raise ValueError(f"non-positive fitted physical parameters: {parameters}")
            return fitted
        except Exception as exc:  # pragma: no cover - exercised with monkeypatch tests
            last_error = exc
            fit_warnings.append(f"attempt {attempt + 1} failed: {type(exc).__name__}: {exc}")

    message = "physics-informed GP fitting failed"
    if last_error is not None:
        message = f"{message}: {last_error}"
    raise PhysicsGPFitFailure(message)


def train_exact_gp(
    mll: ExactMarginalLogLikelihood,
    model: SaturationResidualGP,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    settings: PhysicsInformedGPSettings,
) -> None:
    """Train the custom GPyTorch model with a short deterministic Adam loop."""

    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    for _step in range(settings.max_training_steps):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite physics GP loss: {float(loss)}")
        loss.backward()
        optimizer.step()


def physics_training_tensors(
    records: Sequence[Stage1ExperimentRecord],
    bounds: Range,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert Stage 1 records into ExactGP float64 tensors."""

    if len(records) < 2:
        raise ValueError("physics-informed GP fitting requires at least two observations")
    for record in records:
        if not bounds.contains(record.dose_s):
            raise ValueError(f"dose outside GP bounds: {record.dose_s}")
    train_x = dose_tensor([record.dose_s for record in records])
    train_y = torch.tensor([record.observed_growth for record in records], dtype=torch.double)
    return train_x, train_y


def initialize_physical_parameters(
    records: Sequence[Stage1ExperimentRecord],
    bounds: Range,
    settings: PhysicsInformedGPSettings,
) -> tuple[float, float]:
    """Return robust positive initial values for g_inf and k."""

    max_growth = max((record.observed_growth for record in records), default=0.0)
    initial_g_inf = max(settings.min_initial_g_inf, max_growth * 1.10)
    positive_doses = [record.dose_s for record in records if record.dose_s > 0.0]
    dose_span = max(bounds.max - bounds.min, settings.min_initial_k)
    initial_k = 1.0 / max(min(positive_doses, default=dose_span), settings.min_initial_k)

    target = 0.632 * initial_g_inf
    crossed = [
        record.dose_s
        for record in records
        if record.dose_s > 0.0 and record.observed_growth >= target
    ]
    if crossed:
        initial_k = 1.0 / max(min(crossed), settings.min_initial_k)
    initial_k = max(settings.min_initial_k, initial_k)
    return initial_g_inf, initial_k
