# Repository Guidance

This repository is an educational tutorial, not a predictive chemistry package.

- Keep the virtual laboratory deterministic when seeds and noise settings are fixed.
- Do not claim that the simulator predicts real HfO2/MoS2 area-selective deposition.
- Keep LLM reasoning records concise. Store only short rationales, never hidden chain-of-thought.
- Prefer tests that exercise scientific behavior and safety guards over brittle snapshots.
- Run `ruff format`, `ruff check`, `mypy`, and `pytest` before considering changes complete.

