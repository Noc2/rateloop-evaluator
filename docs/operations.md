# Local operations

Use a dedicated account and an encrypted disk. Keep model files, state, credentials, exports and backups outside the source checkout. The learning store, cache and outbox encrypt their payloads; model weights and exported reports rely on the encrypted volume. Protect encryption and signing keys separately in your backup system. Loss of the encryption key makes records unrecoverable.

## Installation choices

- **Mac:** native Python 3.12 and `.[model]`, with `--device mps`. Edit absolute paths in `deploy/ai.rateloop.evaluator.plist`, place it in the service user's LaunchAgents directory and load it using `launchctl bootstrap gui/UID PATH`. Native MPS inference and training were verified on the maintainer's M5 Max; the launchd template itself is an installation example, not a completed enterprise installation.
- **Linux:** install under `/opt/rateloop-evaluator`, create a dedicated `rateloop-evaluator` user, provision models under an allowed read-only path, and configure `deploy/rateloop-evaluator.service`. CPU is the default. CUDA requires a matching PyTorch/driver installation and separate hardware acceptance tests.
- **Container:** `docker build -f deploy/Dockerfile -t rateloop-evaluator:local .` builds the application. Mount private state and provisioned models at their configured absolute paths. Expose an authenticated TLS listener explicitly for cross-container/network access. Do not mount the Docker socket or bake keys/weights into the image. This CPU-oriented example was not used for the Mac MPS measurements.

The service binds loopback by default and rejects browser origins. Remote serving requires both `--tls-cert` and `--tls-key`; put the worker behind customer network controls. Grant only needed token roles. The runtime intentionally has no cloud model fallback, proxy inheritance, automated package/model update or training-report upload. Enforce no-outbound-network policy at the OS/container boundary for strict offline operation; software environment flags are not a firewall.

Provision public checkpoints before disconnecting networking. Package versions and full model SHAs are recorded in verification evidence. Maintain a tested environment lock for each OS and device instead of assuming one GPU dependency lock is portable. Install GLiClass `.[compare]` in a separate virtual environment: its Transformers 5 dependency conflicts with GLiNER's Transformers 4 requirement.

`requirements-macos-py312.lock` records the exact Python package versions from the verified native Mac environment. On that platform, install with `pip install -c requirements-macos-py312.lock -e '.[model,test]'` to reproduce those versions. It is a version snapshot, not a hash-verified universal lock; select and verify wheels separately for an offline enterprise installation.

## Upgrades and recovery

1. Pause connected RateLoop evaluation and stop the local worker. Preserve a consistent encrypted state/key backup and the exact current bundle.
2. Build a separate environment and verify the replacement package/checkpoint, license, tokenizer and files. Run offline inference, training/reload where used, and customer regression checks.
3. Create a new immutable bundle ID. A new tokenizer, quantization, weights, question definition or language scope requires fresh calibration and validation. Do not overwrite an active model directory.
4. Start in shadow mode, verify connection metadata and human-review behavior, then use explicit local promotion only after quality gates pass. The current RateLoop integration remains advisory.
5. Revert using a still-valid earlier environment and bundle. Registry rollback rejects retired models and expired selective evidence; returning to shadow or human review is the safe recovery if no eligible bundle remains.

Back up only with workers/training stopped. Test restoring a copy onto a separate encrypted location, then verify keys, signatures, file hashes and current consent. Drain the connector outbox after connectivity returns; old receipts outside RateLoop's ingest window are not evidence of a new evaluation. Monitor rejected receipts, repeated abstentions, grant expiry, drift and audit completion rather than logging customer text.

## Capacity and limits

One model worker serializes inference and returns 429 when busy; callers retry with the same idempotency key. Deadlines prevent late approval but do not interrupt a GPU kernel. The initial encrypted whole-state learning store targets a single-host pilot. Its per-operation cost grows with retained data; benchmark full HTTP latency with the intended dataset and concurrency before scaling. Keep administrative training out of the inference process and schedule it separately if both share one GPU.

The signed model registry and evaluation gates do not replace authentication of reviewers, representative sampling, customer consent, host hardening or a customer acceptance exercise. Full offline human-review UI, federated aggregation, multimodal judgments and hosted shared training are subsequent products; no private data is transferred to implement them implicitly.
