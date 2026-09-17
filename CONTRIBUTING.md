# Contributing

Use Python 3.12 and Node 24. Install `.[test]`, run `python scripts/check_contracts.py`, then `python -m pytest -q`. Tests do not download models. Real-model tests are opt-in and use synthetic data; see [verification](docs/verification.md).

Keep model/backend changes separate from protocol and consent changes. Changes to a shared invariant must test every consumer against the same boundary cases. Regenerate schemas with `python scripts/check_contracts.py --write`; maintain Python, TypeScript and RateLoop SDK parity.

Use public or synthetic fixtures only. Never commit API keys, private datasets, trained customer weights or machine-specific state. New model versions require pinned revisions, verified licenses, offline inference and training tests, and fresh calibration. Preserve abstention and human review when evidence is missing.

Contributions are provided under Apache-2.0. Upstream software, weights and data retain their own licenses and provenance.
