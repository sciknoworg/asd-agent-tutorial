# Repository Guidance

This repository is an educational tutorial, not a predictive chemistry package.

- Keep the virtual laboratory deterministic when seeds and noise settings are fixed.
- Do not claim that the simulator predicts real HfO2/MoS2 area-selective deposition.
- Keep LLM reasoning records concise. Store only short rationales, never hidden chain-of-thought.
- Prefer tests that exercise scientific behavior and safety guards over brittle snapshots.
- Keep oracle calculations retrospective. Optimizers, stopping rules, acquisition functions, and
  LLM prompts must consume optimizer-facing records only.
- Execute hybrid experiments only by immutable BO candidate ID. Validate candidate ownership,
  hard bounds, budget, and duplicate status independently of any LLM decision.
- Never make live API calls in tests or default smoke profiles. Live Responses API use requires an
  explicit flag plus environment credentials.
- Use named, persisted seeds for simulator state, measurement noise, initialization, BO, and LLM
  behavior. Paired comparisons must reuse matched streams across methods.
- Export figure source data beside generated plots. Prefer PNG plus SVG/PDF for publication output.
- Preserve legacy public interfaces and commands when extending BO functionality.
- Manual laboratory support is plan export and validated measurement ingestion only; do not add
  autonomous reactor control.
- Run `ruff format`, `ruff check`, `mypy`, and `pytest` before considering changes complete.
- Keep manual laboratory handoff code human-operated. Do not add autonomous reactor
  control, instrument drivers, or real-lab safety claims without an explicit new task
  and qualified human review.
- Keep BO oracle outputs evaluation-only. Optimizers, LLM prompts, and manual lab
  plans must not receive hidden simulator parameters or oracle-only answers.
