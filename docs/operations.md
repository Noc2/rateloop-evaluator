# Local operations

Use a dedicated account and an encrypted disk. Keep model files, state, credentials, exports and backups outside the source checkout. The learning store, cache and outbox encrypt their payloads; model weights and exported reports rely on the encrypted volume. Protect encryption and signing keys separately in your backup system. Loss of the encryption key makes records unrecoverable.

## Installation choices

- **Mac:** native Python 3.12 and `.[model]`, with `--device mps`. Edit absolute paths in `deploy/ai.rateloop.evaluator.plist`, place it in the service user's LaunchAgents directory and load it using `launchctl bootstrap gui/UID PATH`. Native MPS inference and training were verified on the maintainer's M5 Max; the launchd template itself is an installation example, not a completed enterprise installation.
- **Linux:** install under `/opt/rateloop-evaluator`, create a dedicated `rateloop-evaluator` user, provision models under an allowed read-only path, and configure `deploy/rateloop-evaluator.service`. CPU is the default. CUDA requires a matching PyTorch/driver installation and separate hardware acceptance tests.
- **Container:** `docker build --target runtime -f deploy/Dockerfile -t rateloop-evaluator:local .` builds the application. Mount private state and provisioned models at their configured absolute paths. Expose an authenticated TLS listener explicitly for cross-container/network access. Do not mount the Docker socket or bake keys/weights into the image. The pinned CPU image contains no CUDA libraries. Its default final stage is the outbound hosted worker described below; `--target runtime` retains the standalone CLI entrypoint.

The service binds loopback by default and rejects browser origins. Remote serving requires both `--tls-cert` and `--tls-key`; put the worker behind customer network controls. Grant only needed token roles. The runtime intentionally has no cloud model fallback, proxy inheritance, automated package/model update. Training reports upload only through the explicitly enabled website training runner described below. Enforce no-outbound-network policy at the OS/container boundary for strict offline operation; software environment flags are not a firewall.

Provision public checkpoints before disconnecting networking. Package versions and full model SHAs are recorded in verification evidence. Maintain a tested environment lock for each OS and device instead of assuming one GPU dependency lock is portable. Install GLiClass `.[compare]` in a separate virtual environment: its Transformers 5 dependency conflicts with GLiNER's Transformers 4 requirement.

`requirements-macos-py312.lock` records the native Mac package versions, including subsequent verified security dependency updates. On that platform, install with `pip install -c requirements-macos-py312.lock -e '.[model,test]'` to reproduce those versions. It is a version snapshot, not a hash-verified universal lock; select and verify wheels separately for an offline enterprise installation.

### Native Mac background-service paths

Install the service's source, virtual environment, dependencies, models and private state under a dedicated directory such as `~/Library/Application Support/RateLoop Evaluator/`. Dependencies must also live outside protected `Documents`, `Desktop` and `Downloads` directories: a `.pth` file, editable install or symlink back to a development checkout can reintroduce protected-path access even when the service executable itself is elsewhere. A successful interactive terminal import does not establish that a background LaunchAgent can read the same paths.

Before starting the service, inspect that interpreter's `sys.path` and the imported package's `__file__`; resolve dependency symlinks and confirm they stay within the intended installation. Preserve the exact package versions and verify inference after relocation. Treat a launchd `running` state as process existence, not readiness: require a fresh authenticated worker heartbeat in RateLoop. If no heartbeat arrives, inspect process activity and startup diagnostics before changing workspace permissions. For diagnosis, point `StandardOutPath` and `StandardErrorPath` at owner-only files in a private directory, keep logs bounded, and share only sanitized errors.

On 27 September 2026, native runtime `e9a7980ad0b5d91e5a6c524d78e5f4b925994f35` remained alive without sending a heartbeat while its import stack waited in directory access. Its dependency path led into `Documents`. Copying the same dependencies into Application Support, removing the development editable-path loader and restarting the same launchd label restored production heartbeats. Credentials, keys, configuration and encrypted state were byte-checked unchanged before startup; the runtime source and hosted services were unchanged. This records recovery of startup/readiness, not completion of a connected training job.

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

## Bounded guiding examples

Custom English/German binary questions can include two or a few labeled examples without a training run:

```python
from rateloop_evaluator.templates import custom_text_evaluation

template = custom_text_evaluation(
    "en", "Does the text state an explicit numeric budget?", "Yes", "No",
    examples=[
        {"text": "The project budget is EUR 750.", "labelId": "approved"},
        {"text": "The project has 8 stages; its budget is undecided.", "labelId": "rejected"},
    ],
)
```

Use the declared answer IDs, not their display wording. The portable question field is `examples` with strict `text` and `labelId` fields: at most four entries, each 1–600 UTF-16 units after trimming, at most 1,600 combined. Invalid Unicode, disallowed control characters and undeclared labels are rejected. Omitted and empty examples preserve the old serialized template and commitment. Nonempty examples, their order, wording or labels change the template commitment; retain old templates for queued work and obtain the exact current processing scope for the new template. Calibration from the old question is never transferred.

Examples use the pinned GLiNER processor's existing example pairs. The same ordered question, label IDs, descriptions and examples feed inference and optimizer inputs; classification schema augmentation is disabled to preserve the immutable rubric. Targets bind to the complete question schema, including when question IDs share a prefix. Adapter version 1 and complete mutually exclusive softmax scores are unchanged. Input, evidence, question, labels and examples all count toward the existing 512-token custom-task budget. A combined input above the limit is rejected or abstains; it is never silently shortened. GLiClass comparison explicitly rejects questions with examples instead of ignoring them.

Keep demonstrations separate from data used to measure performance. Dataset import, training and frozen comparison reject a dataset text that repeats an example after whitespace normalization, including older snapshots checked at comparison time. This catches exact normalized text, not paraphrases or related sources; review those and keep their source groups together. Try the changed rubric against held-out examples before adopting it: [the recorded bilingual experiment](verification.md#bilingual-diagnostics-and-guiding-examples--27-september-2026) found no overall improvement from two examples and one email-case regression.

## Compare the pinned Decide challenger

Decide is an experimental local checkpoint option. It is **not the hosted default**, a training-base replacement or a promotion recommendation. Provision explicitly into a new empty directory in an online process, then compare in a separate offline process:

```sh
rateloop-evaluator provision --checkpoint decide --model-dir /private/models/gliner25-decide
rateloop-evaluator diagnose --model-dir /private/models/gliner25-decide --device cpu \
  --output /private/reports/decide-diagnostics.json
```

This pins `fastino/GLiNER2.5-multi-Decide` to `6bc1d43d201b0691e733626389af8c57eea3ea68`, verifies its Apache-2.0 declaration and records every artifact hash. The default `provision` and hosted provisioning continue to use the existing public base. Run the base and challenger sequentially in separate processes, avoiding simultaneous resident checkpoints. Diagnostics do not import training data, create permission grants, register a bundle, activate a model or create a paid service.

The current custom hosted capability and private-training recipe still require the reviewed public base. Any future supported replacement needs a distinct model identity, fresh task-specific calibration and hosted memory/latency acceptance. The [measured challenger](verification.md#bilingual-diagnostics-and-guiding-examples--27-september-2026) performed worse on the small authored suite; keep the existing model. Local Mac memory and timing measurements do not establish compliance with the hosted 2.5 GB cap or its monthly budget.

## Outbound website worker

The Alpha website accepts its explicitly configured RateLoop-operated worker. That worker can run on an always-on Linux CPU service or a connected Mac. Standalone customer-local operation is supported; website admission remains controlled by the deployed application.

Use a dedicated private state directory initialized with the actual workspace ID. Provision and register the overall-approval request for each intended language, then import the exported registration into the workspace. The connector's agent and version IDs must match the selected website integration. The workspace credential needs `evaluation:read` and `telemetry:write`. For AI + human, the website creates the authorized human review before worker inference; AI-only cases create no human review.

```sh
rateloop-evaluator --state-dir /private/evaluator init --workspace WORKSPACE_ID
rateloop-evaluator --state-dir /private/evaluator register --activate --model-dir /private/models/gliner25 --request /private/approval-request.json
rateloop-evaluator --state-dir /private/evaluator export-registration --bundle-id BUNDLE_ID --request /private/approval-request.json --output /private/registration.json
rateloop-evaluator --state-dir /private/evaluator worker --config /private/connector.json --worker-id pilot-mac --bundle-id BUNDLE_ID --device mps --once
```

The private connector file follows [the connector configuration](connector.md) and uses `https://www.rateloop.ai` for the Alpha. Enable AI processing for the selected operator's hardware before submitting a case. Enable private learning separately, before collecting training cases. Shared data and public-weight permissions remain independent.

Choose Human, AI or AI + human before submitting the website case. Human-only creates no evaluator job. AI-only returns its advisory rating when the worker completes, without requiring reviewers; low-confidence or uncalibrated results remain uncertain. AI + human preserves independent review by withholding the AI result until the human answer freezes. The model's local shadow/selective qualification is separate from this case-level choice. Selecting AI never grants training permission or changes a required human review already in progress.

The worker binds `reviewMode` from the authenticated claim to the content response. An omitted field means the earlier AI + human behavior. AI-only requires an explicit `ai` choice, `audit: null` and `retainForTraining: false`; its raw inputs are not retained for training and its results cannot be imported or submitted as human training labels. Upgrade the worker before selecting AI-only: an older worker correctly refuses a job with no mandatory blinded audit.

Remove `--once` for continuous processing. The worker polls every five seconds, uses a 120-second fenced job lease and renews it every 30 seconds during inference. It refreshes execution consent before scoring and before releasing a receipt. Explicit case tombstones purge local examples, pending metadata and dependent model eligibility when synchronized. An HTTP 401/403 immediately pauses this connector and expires its inference and dataset execution leases. Fresh validated responses from each permission endpoint are required to resume; an endpoint scope failure does not revoke owner consent or retire trained lineage. Explicit withdrawal, deletion and changed immutable permission remain permanent. Legacy grants without renewable leases are revoked and require new owner-issued identities. This never restores a grant already revoked by an older worker: withdraw and explicitly grant fresh affected website consents. A credential failure is not a blanket data-deletion instruction. A permanent receipt rejection fails the website job with a content-free HTTP error code instead of repeatedly renewing its lease. Fix the integration and submit a new case; network failures, rate limits and server errors still retry the same persisted receipt. An owner workspace-deletion notice processes the explicit case IDs returned by the server and retires execution permissions. Local-only cases not identified by that notice require the owner's `delete-case` command and backup-retention process.

Durable consent issue/revocation timestamps and execution-lease issue timestamps tolerate a server clock at most five seconds ahead of the local response-receipt time. The local execution deadline is the earliest of the server lease expiry minus five seconds, receipt time plus 900 seconds, and any finite consent expiry minus five seconds. Leases must still have a positive duration of at most 900 seconds; expired or nearly expired authorizations grant no execution. Wire timestamps, consent hashes and stable lineage remain unchanged, and training collection still compares the original server consent/case timestamps without tolerance. Legacy immutable grants with up to 24-hour offline lifetimes retain their strict timestamp checks and original expiry; this change does not migrate or extend them. Keep both clocks synchronized; this is a small clock-skew allowance, not protection against host-clock rollback.

When the website reports off or paused, durable consent records remain visible but no execution lease is issued. The worker expires only the local renewable authorization, retains unchanged consent identities and training lineage, and continues metadata-only presence. A later shadow-mode response can renew that same unrevoked consent only with a fresh valid scoped lease. Missing, expired, locally revoked or explicitly withdrawn consents remain revoked, including during a pause. Legacy independent finite learning grants retain their original expiry. For a volume affected by the earlier missing-lease pause bug, existing revocations are never silently undone: the owner must withdraw the affected old website consents and explicitly grant fresh ones before resuming.

The worker imports authorized labels for its exact frozen binary questions about once a minute. Blind human responses frozen before result release remain eligible after reveal; later exposed judgments do not. There is no background training: an operator uses the commands in [the learning guide](learning.md), inspects the holdout results, registers a new immutable bundle and imports its registration in RateLoop. Training, calibration, held-out scoring and inference share a process-level lock for the private state. A busy worker refreshes permissions but claims no jobs until training finishes. A competing operator command reports busy and can be retried; it never silently starts a second model operation. Registration and metadata export create a candidate without activation. Explicitly promote the reviewed candidate, authorize its exact bundle and rubric in RateLoop, then restart the worker with its `--bundle-id`. Retain earlier configured bundle IDs while their queued jobs finish: each job keeps its original activated model and current consent checks. Rollback or removing an ID from the worker can stop a candidate, but never substitutes a different model into a committed job. Repeat the website case to verify the returned provenance names the candidate. Earlier bundles remain rollback choices only while their data lineage remains authorized.

Rollback requires matching choices on the Mac and the website. Stop or drain the worker, refresh connected permissions, and restore the exact previous local deployment before selecting that bundle for future website cases. The local command is `rateloop-evaluator --state-dir /private/evaluator rollback --template-commitment TEMPLATE_HASH --language en --expected-bundle-id PREVIOUS_BUNDLE_ID`. It shares the model execution lock and checks the expected previous bundle, artifact hashes, lineage and evidence before changing activation. A wrong target or retired model leaves the current deployment unchanged. Restart the worker with the restored bundle ID, select that same model on the website, and verify a new case's provenance. Changing only the website choice cannot activate a local model; a worker fails closed on that mismatch.

The synthetic acceptance operator combines the permission refresh, execution lock and checked local rollback: `python scripts/alpha_e2e_operator.py --config /private/operator.json rollback --language en --bundle-id PREVIOUS_BUNDLE_ID`. It returns the restored bundle, template, language and mode as metadata. The browser harness separately changes the website choice. A `worker --once` or `worker-once` command can return a failed or idle state without a nonzero exit code; callers must inspect its returned state and verify the expected job and model IDs before treating the case as completed.

Connected workers validate and warm every configured model before reporting presence. Public and private checkpoints warm serially in one bounded checkpoint cache and reload on demand for their pinned jobs. Switching releases the previous model before loading another. On Linux, a scoped read-only checkpoint cache hint releases unused file copies after load/unload; model tensors and files remain intact. See the [CPU capacity measurements](verification.md#hosted-private-model-cpu-capacity--28-september-2026). English and German registrations with the same immutable local artifacts share one model instance. A warm worker reports presence while the workspace is off or paused, so an owner can see it before enabling AI; it still cannot claim content without current authorization. Connected workers report ready or busy while polling and running inference. Explicit connected training reports training immediately and refreshes it every 30 seconds after checking current consent. A unique operation ID prevents a polling worker or an older training process from clearing a newer training status; presence expires after 120 seconds without a matching refresh. Presence conveys availability only and never authorizes processing or learning. After a failed final status update, the last report remains visible only until that timeout.

The idempotent presence update retries at most three times, after 100 ms and 300 ms, only when HTTP 503 explicitly reports `database_coordination_busy` with `retryable: true`. Other server failures, transport failures and authentication refusals are not retried by this path. This does not change retry behavior for receipts or other connector writes.

The acceptance operator uses `rateloop_evaluator.presence.connected_training(connector, worker_id=..., model_bundle_ids=[...])` around its complete in-process train/calibrate/register/score sequence. Production operator scripts can use the same context with their explicitly configured `RateLoopConnector`, the installed worker's identity, and already registered source bundles. The context acquires the shared execution lock before reporting training, yields a `check()` function to call between stages, renews consent during the operation, and attempts a matching ready report in `finally`. Invoke Python training and CLI functions within that process; do not hold the parent lock while starting separate training subprocesses. The ordinary local `train` command has no connector or unsolicited network traffic; while it holds execution, a separately running connected worker reports busy rather than training.

Progress, fencing tokens, results and receipt acknowledgments persist encrypted. A restart renews the saved lease or drops a superseded claim, and retries use the same input and receipt commitments. A Mac that sleeps appears offline; queued cases wait or return a recoverable failure after the server's bounded attempts. An absent worker cannot approve a case or silently switch it to human review. Use an awake, connected account for an unattended pilot; later move the same worker to an always-on machine if needed.

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


## Always-on hosted CPU worker

Build the default `hosted` stage of `deploy/Dockerfile` for `linux/amd64`. Python, CPU PyTorch and dependencies are pinned in the image and `deploy/constraints-cpu.txt`. The image contains software only: public weights are explicitly provisioned onto a private persistent volume and raw content, credentials, private weights and keys must never enter Git or image layers.

Use one replica, one existing workspace connector, one private volume mounted at `/data`, and one CPU. An optional native Chat pool uses separate case-specific tenant grants as described below. The initial base-model pilot uses a **2,500,000,000-byte RAM cap**; budget and workload must be checked before deployment. CPU time and input length determine throughput. This is a serialized pilot. Optional private training requires the explicit configuration and separately authorized destination described below, plus a measured burst-memory limit; the initial inference cap is not a training capacity claim. Keep public networking disabled: the only listener is a metadata-only `/healthz` for the platform's private health check on port 8080. It has no inference, administration or credentials endpoint.

Set private service variable `RATELOOP_HOSTED_CONFIG_JSON` to this JSON shape, replacing all example identities and the credential:

```json
{
  "schemaVersion": "rateloop.hosted-worker.v1",
  "workspaceId": "WORKSPACE_ID",
  "workerId": "WORKER_ID",
  "modelDir": "/data/models/gliner25",
  "stateDir": "/data/state",
  "bundles": [
    {"language": "en", "modelBundleId": "hosted-gliner25-en-v1"},
    {"language": "de", "modelBundleId": "hosted-gliner25-de-v1"}
  ],
  "connection": {
    "baseUrl": "https://www.rateloop.ai",
    "apiKey": "PRIVATE_WORKSPACE_CREDENTIAL",
    "apiKeyId": "API_KEY_ID",
    "agentId": "AGENT_ID",
    "agentVersionId": "AGENT_VERSION_ID",
    "metadataUploadEnabled": true
  },
  "pollSeconds": 5,
  "healthPort": 8080
}
```

The entrypoint changes only `/data` ownership when the platform mounts it as root, then drops all supplementary groups and runs as UID/GID 10001 before opening configuration, provisioning or inference. It writes configuration with owner-only permissions and removes the secret environment variable before executing the runtime. `/data/state` retains encryption/signing keys, encrypted state and fencing/outbox progress across restarts. Restart never changes the workspace, worker, agent, model or bundle identity. A credential can rotate within that identity; fresh server authorization is still required.

For first installation, explicitly set `RATELOOP_PROVISION_MODEL=1`. The entrypoint runs **a separate provisioning process** pinned to the reviewed `MODEL_REVISION`, then starts a new offline runtime process. Existing model files are validated and reused; malformed or tampered files fail startup rather than downloading replacements. Clear the provisioning flag after successful installation. Alternatively provision the same pinned weights onto the volume beforehand and omit the flag. Inference and training never perform provisioning.

The standalone lifecycle commands are `python -m rateloop_evaluator.hosted prepare|bootstrap|run --config /private/hosted.json`. `prepare` is the only command that can download a model. `bootstrap` validates local weights, preserves existing keys and registrations, and writes app-registration metadata to `/data/state/registrations/en.json` and `de.json`. Bootstrap does not authorize AI processing or learning. Import the matching registrations and configure the workspace-bound server credential before enabling AI use.

### Native Chat case pool

The retained hosted process can also poll the application's dedicated native Chat queue.
Add `nativeChatPool` to its private configuration with the canonical `baseUrl`
`https://www.rateloop.ai` and a distinct server-only `secret` of at least 32 random
characters. Configure the same secret and the two actual public custom-text
registrations in the application. Keep the existing connector, keys, bundle IDs,
volume and destinations intact. Never reuse a customer workspace credential as
the pool transport secret.

The pool receives only jobs joined to persisted native Chat mappings and exact
tenant/agent/version/key/model destinations. The application checks current
AI-use consent and processing state on claim, content release, renewed lease and
receipt. A separate persisted rating allowance is reserved before plaintext is
released. The worker validates the frozen case commitments and actual pinned
public artifacts; it grants one case AI use until its lease expires, with no
training or sharing rights. Raw answer/context remains in memory only. The
ephemeral encrypted case metadata/result cache is destroyed after inference,
and a transient receipt retry retains only its receipt and fencing metadata until
the original lease expires. A permanent receipt rejection discards the result and
reports a content-free terminal failure; temporary failure-report outages retain
only that bounded failure intent. Other tenants can then continue processing.
Both inference queues release saved completion progress only after the application
explicitly acknowledges `completed: true`; malformed success responses retry the
original receipt and fencing token without another inference. Both queues retain
a nonretryable failure intent until the application returns literal `retrying: false`
or the original lease is lost. Missing or ambiguous acknowledgments retry only
content-free failure metadata, without renewing the lease or rerunning inference.
The retained worker persists this intent across restarts and case deletion purges it.
Inference-core errors use the same terminal failure route immediately, with a
fixed error category. Backend exception details and customer text are never sent;
ephemeral case files are destroyed even when inference fails.
Process restarts recover through the durable application lease and bounded
attempt count, rather than a second hidden inference queue.

Both queues execute serially through the existing one-checkpoint cache. No second
listener, model download, private-model export, inferred calibration or paid model
fallback is added. The retained workspace health check stays independent of
the native queue during application releases; native availability expires with
its separate application heartbeat. A warm pool advertises availability only,
and does not authorize first-send processing. Short English/German outputs may
produce an experimental binary classifier label and raw score. Longer inputs
abstain at the existing 512-token limit. These results are not calibrated
correctness probabilities or permission to replace independent human review.

On 2026-10-02, the implementation in `757e08b` passed 560 repository tests
(five optional checks skipped). An offline check using the exact public
`fastino/gliner2.5-multi-v1` revision
`235cf92d6d4318da9bfca0d08975c8fa7250d13b` exercised the default English and
German relevance questions through the native case-grant inference path with
network connection attempts denied. All four short synthetic cases returned
valid uncalibrated receipts, with 40–73 ms warm inference on the operator's CPU.
The two intentionally irrelevant answers were incorrectly labelled approved;
their raw scores were also high. This is evidence of transport and rubric
compatibility, and a concrete reason to keep the default classifier
experimental. It establishes neither general relevance accuracy nor correctness,
confidence calibration or hosted latency. No synthetic input or credential is
retained in this repository. Live release acceptance must additionally verify a
persisted native case on the retained hosted worker.

For an explicit first-installation handoff, set server-only `RATELOOP_HOSTED_EXPORT_REGISTRATIONS=1`. After bootstrap, the entrypoint writes one `RATELOOP_HOSTED_REGISTRATION_V1 ` log line per language, followed by the exact registration JSON stored on that volume. These contain registration metadata only; no credential, workspace configuration, input or private weights are printed. Retrieve those actual cloud registrations from private provider logs and import them before enabling AI use, then clear the export flag. Ordinary startup emits none. A placeholder credential may be used for this isolated bootstrap; replacing it does not rewrite the actual model activation evidence. Workspace, worker, agent/version and bundle identities must already be final. Health remains unavailable until a valid scoped credential connects successfully.

Readiness stays HTTP 503 until all model warmups finish and the worker completes a successful authenticated poll and heartbeat. It returns only `{"status":"ready"}` while that connection is fresh; failed polls, stale polling or shutdown restore 503. A paused or off workspace can be healthy because presence itself grants no content access. SIGTERM stops polling cleanly; failed requests use bounded retry delays. A volume/worker lock prevents duplicate processes from sharing the same durable identity.

Before release, run the normal tests and the opt-in container acceptance check:

```sh
docker build --platform linux/amd64 -f deploy/Dockerfile -t rateloop-evaluator:hosted .
python scripts/test_hosted_container.py rateloop-evaluator:hosted /absolute/path/to/provisioned/gliner25
```

That check uses the real local model, no network, and a test-only paused-server transport mounted outside the image. It checks cold/warm readiness, non-root runtime, the exact RAM/CPU cap, no OOM kills, SIGTERM, restart and encryption-key continuity. It is not evidence of a real hosted website case: deployment acceptance must also exercise an authorized case through the live application and verify its persisted result, selected bundle, permissions and blinding behavior. `scripts/measure_cpu.py` separately measures synthetic English/German inference and cgroup memory; emulated amd64 latency on Apple hardware is not a Railway service SLA.

## Optional website training on an operator machine

The hosted container is inference-only by default. To enable always-on private training, set `privateTraining: true` in its protected configuration, explicitly allowlist that exact workspace/key/worker destination on the website, and obtain the separate dataset permission. No Mac checkpoint is copied automatically. New candidates are created on the hosted private volume and can be evaluated there while the Mac is asleep.

Hosted training runs in a fresh CPU process after releasing inference weights. Its signed candidates, encrypted examples, keys and job state remain on that workspace's volume. Successful permission heartbeats keep health current during training. Inference queues while training or comparison runs; this budget configuration does not provide concurrent training and evaluation. After the child exits, the parent rechecks permissions and restores the exact configured bundle allowlist. Model memory is released on process exit, and interrupted jobs retain fenced progress for recovery. A durable one-hour aggregate training/comparison budget per UTC day and a one-hour operation ceiling bound resource use. The parent reserves budget before starting the child and refunds unused time only after it survives cleanup; a crash retains the conservative charge. Activation and rollback are exempt. Training also requires at least 3 GiB of free volume space. Resource exhaustion and overlong operations fail without retry loops. SIGTERM interrupts the child and preserves recoverable progress.

Use `scripts/measure_hosted_training.py` with the reviewed CPU image, the public checkpoint read-only, `--network none`, one CPU and an explicit memory cap before changing hosted capacity. This authored synthetic run measures training, save/reload, frozen comparison and repeated public/private switching. It does not establish general evaluator quality or a monthly price; retained memory, volume usage and training frequency still require monitoring.

An operator-owned machine may instead accept the same bounded website jobs by explicitly opting in:

```sh
rateloop-evaluator --state-dir /private/evaluator worker \
  --config /private/connector.json --worker-id operator-training \
  --bundle-id PUBLIC_BASE_BUNDLE --device mps \
  --allow-training --training-model-dir /private/models/gliner25
```

Both training flags are required. `install-launchd` accepts the same flags when this machine should remain available. Provision and register the pinned public base first, and allowlist this worker/API-key destination in the website. A hosted destination retains its own inference identity and requires separate explicit training admission; an operator machine uses its own configured identity. No model or training code is downloaded; jobs cannot select paths, scripts, dependencies or arbitrary hyperparameters. The website requests the reviewed `rateloop.evaluator.training-recipe.v2`: up to 200 optimizer steps, learning rate 0.0001, a fixed 20% source-group validation split taken only from the frozen training partition, evaluation every 25 steps, and early stopping after three unimproved evaluations. Each label needs at least five independent source groups in both training and validation; small datasets can still be compared but cannot establish a validation-selected candidate. The best validation checkpoint is restored before final held-out comparison. A v2 run must strictly improve validation balanced agreement without reducing either class's recall against the original evaluator, or it produces no registered candidate. Existing queued v1 jobs retain their fixed short recipe and remain explicitly unqualified smoke runs. Pinned upstream GLiNER uses a positive step budget to derive the number of data passes, overriding the configured epoch count; a small training split can therefore be traversed more than once. No private weights leave this machine.

The runner advertises `rateloop.evaluator.training-worker.v1` with `recipeSchemaVersion: rateloop.evaluator.training-recipe.v2` through the dedicated training heartbeat. An ordinary hosted inference heartbeat does not imply training support. A dataset has its own owner-authorized, exact workspace/key/worker/template/base/version/content permission and at most 30 days of retention. A short renewable execution lease is independent of the dataset's retention permission and the current training job. Dataset deletion or permission removal erases that version’s local example and snapshot row copies on the next synchronization and retires derived candidates. The running worker also enforces the durable retention deadline when the website is unavailable; a stopped machine applies it on restart. Hashed holdout assignments remain to prevent later train/test leakage. Existing checkpoint files and backups require separate operator deletion; no instant unlearning is claimed. Cancelling a job stops further optimizer updates and suppresses its result without deleting a still-authorized dataset. Imported examples retain their declared provenance and cannot become blind-human qualification evidence.

Compare/train use the same frozen local snapshot for a website dataset version. A training result includes the unchanged reviewed registration, signed manifest and an unqualified baseline/candidate comparison. It does not activate the candidate. A separate owner-requested activation selects the exact candidate on this runner, makes it available for inference using a separate exact AI-use permission, and retains signed rollback history. The website drains pending evaluations for the task before switching; the runner keeps at most one resident checkpoint, whether public or private. Other authorized queued bundles reload their exact artifact on demand after evicting the previous checkpoint. Rollback names the expected prior bundle. The local switch is applied only after the website acknowledges completion. An interrupted completion retries the same encrypted result and reconciles the acknowledged switch before any new inference, without rerunning training or changing candidate identity. If the optimizer was killed before creating a final manifest, the runner preserves that job's unsigned partial output in `training-interrupted`, then retries an empty output directory against the same frozen snapshot and recipe. Completed manifests are validated and reused; they are never overwritten. Hosted retries remain subject to the durable daily budget. A source revocation during reconciliation retires the candidate and prevents its activation.

Training preserves the pinned base tokenizer assets byte for byte and compares the loaded tokenizer semantics before training, after training and after checkpoint reload. The exported registration must retain the base tokenizer commitment. A completed checkpoint with a different tokenizer identity must not be rewritten under its existing signed candidate ID; fix the runtime and submit a new training job. The trained weights, adapter commitment and calibration still receive their own identities.

Failure reporting sends only `workerId`, `leaseToken` and a content-free `errorCode`. The worker persists that terminal intent until the website acknowledges the same job with `status: failed`. Transport errors, rejected payloads and ambiguous acknowledgements preserve it for retry without retraining or resubmitting a rejected completion. An explicit 404, 409 or 410 releases a missing, superseded or expired lease. This training contract differs from the evaluation job failure contract; do not add its `retryable` field here.

Outbound endpoints are under `/api/assurance/v2/evaluations/training`: `/workers/heartbeat`, `/jobs/claim`, and `/jobs/{id}/content|heartbeat|complete|fail`. Content retrieval uses the fenced `X-Evaluator-Lease` and `X-Evaluator-Worker` headers. Every job binds the exact task and dataset commitment; the dataset digest domain is `rateloop.evaluator.dataset.v1` over `{workspaceId,templateCommitment,provenance,rows}` using RFC 8785. Completion uploads metadata only, never dataset rows, artifact paths or checkpoint bytes.
