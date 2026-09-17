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

## Outbound website worker

Use a dedicated private state directory initialized with the actual workspace ID. Provision and register the overall-approval request for each intended language, then import the exported registration into the workspace. The connector's agent and version IDs must match the selected website integration. The workspace credential needs `evaluation:read` and `telemetry:write`; the website creates the authorized review before worker inference.

```sh
rateloop-evaluator --state-dir /private/evaluator init --workspace WORKSPACE_ID
rateloop-evaluator --state-dir /private/evaluator register --model-dir /private/models/gliner25 --request /private/approval-request.json
rateloop-evaluator --state-dir /private/evaluator export-registration --bundle-id BUNDLE_ID --request /private/approval-request.json --output /private/registration.json
rateloop-evaluator --state-dir /private/evaluator worker --config /private/connector.json --worker-id pilot-mac --bundle-id BUNDLE_ID --device mps --once
```

The private connector file follows [the connector configuration](connector.md) and uses `https://www.rateloop.ai` for the Alpha. Enable AI processing for the selected operator's hardware before submitting a case. Enable private learning separately, before collecting training cases. Shared data and public-weight permissions remain independent.

Remove `--once` for continuous processing. The worker polls every five seconds, uses a 120-second fenced job lease and renews it every 30 seconds during inference. It refreshes execution consent before scoring and before releasing a receipt. Explicit case tombstones purge local examples, pending metadata and dependent model eligibility when synchronized. A credential failure stops execution; it does not imply a blanket data-deletion instruction. An owner workspace-deletion notice processes the explicit case IDs returned by the server and retires execution permissions. Local-only cases not identified by that notice require the owner's `delete-case` command and backup-retention process.

The worker imports authorized overall human labels about once a minute. Blind human responses frozen before result release remain eligible after reveal; later exposed judgments do not. There is no background training: an operator uses the commands in [the learning guide](learning.md), inspects the holdout results, registers a new immutable bundle and imports its registration in RateLoop. Training, calibration, held-out scoring and inference share a process-level lock for the private state. A busy worker refreshes permissions but claims no jobs until training finishes. A competing operator command reports busy and can be retried; it never silently starts a second model operation. Drain the worker before a managed model change. Authorize the new bundle explicitly, then restart the worker with its `--bundle-id`. Repeat the website case to verify the returned provenance names the candidate. Earlier bundles remain rollback choices only while their data lineage remains authorized.

Progress, fencing tokens, results and receipt acknowledgments persist encrypted. A restart renews the saved lease or drops a superseded claim, and retries use the same input and receipt commitments. A Mac that sleeps appears offline; queued cases wait or return a recoverable failure after the server's bounded attempts. Human review is never satisfied by an absent worker. Use an awake, connected account for an unattended pilot; later move the same worker to an always-on machine if needed.

For a native macOS login service, install the reviewed package without editable mode into a persistent virtual environment outside Documents, Desktop, Downloads and cloud-synced folders. Keep the service's configuration, state and model files outside those folders too. A terminal's access does not establish access for a background service. Use the user's Application Support directory, for example:

```sh
EVALUATOR_RUNTIME="$HOME/Library/Application Support/RateLoop Evaluator/runtime"
python3.12 -m venv "$EVALUATOR_RUNTIME"
"$EVALUATOR_RUNTIME/bin/python" -m pip install "/absolute/path/to/reviewed/rateloop-evaluator[model]"
"$EVALUATOR_RUNTIME/bin/rateloop-evaluator" --state-dir /private/evaluator install-launchd --config /private/connector.json --worker-id pilot-mac --bundle-id BUNDLE_ID --device mps --load
```

This creates an owner-only LaunchAgent file and loads it into the current user's session. Its arguments contain a protected configuration path, not an API credential. It never overwrites an existing plist. The command returns its label and absolute path; stop it with `launchctl bootout gui/$(id -u) /absolute/path/to/the.plist` before an upgrade. Installations are per logged-in account, not system-wide daemons. Stop the old worker before changing paths or model bundles; the same worker identity cannot run twice against one state directory.

Inspect the returned service label with `launchctl print gui/$(id -u)/LABEL`; restart it with `launchctl kickstart -k gui/$(id -u)/LABEL`. A registered service or assigned PID alone is not proof that the worker can process a case: submit a synthetic case and verify its completed result and expected model bundle after installation. A durable pilot needs its own retained workspace, scoped credential, registered model and explicit AI-use consent; do not reuse the disposable acceptance workspace after cleanup. Private learning requires its separate consent before new cases are collected.

On 2026-09-17, commit `7ed241ad314f69da9a784f47f2a4d0642995db05` passed actual macOS launchd bootstrap, worker-lock acquisition, restart with a new PID, stop and lock release, private plist/configuration permissions, refusal to overwrite an existing plist, and complete service/state cleanup. This check used a temporary runtime outside Documents and a dummy credential directed only at a non-listening loopback port; it verifies service lifecycle, not hosted inference or an unattended deployment. The same runtime launched from the maintainer's Documents checkout blocked while Python read its environment, before worker startup, despite launchd reporting a running PID. No macOS privacy permissions were changed to complete the check.
