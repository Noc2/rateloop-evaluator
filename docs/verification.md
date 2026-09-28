# Local model verification

This maintained record describes dated implementation checks for
`rateloop-evaluator` 0.1.0, including the [27 September bilingual diagnostic comparison](#bilingual-diagnostics-and-guiding-examples--27-september-2026).
It verifies execution, not suitability for replacing human ratings. No customer content was used.

## Hardware and runtime

Apple M5 Max MacBook Pro, 18 CPU cores, 128 GiB unified memory, arm64,
macOS Darwin 25.6.0; Python 3.12.14, PyTorch 2.14.0, FP32, MPS. Other desktop
workloads were not disabled. GPU access required running the verification
outside the development filesystem sandbox.

The primary runtime used GLiNER2 2.0.0, Transformers 4.57.6, tokenizers 0.22.2,
huggingface-hub 0.36.2, PEFT 0.21.0, protobuf 6.33.6 and NumPy 2.5.3.
The checkpoint was
[`fastino/gliner2.5-multi-v1`](https://huggingface.co/fastino/gliner2.5-multi-v1/tree/235cf92d6d4318da9bfca0d08975c8fa7250d13b),
commit `235cf92d6d4318da9bfca0d08975c8fa7250d13b`, Apache-2.0.
It contains 287,355,159 parameters, complete encoder configuration and tokenizer
assets. Runtime startup verifies every declared file hash, rejects undeclared
files and symlinks, and loads from the provisioned directory.

The optional comparison runtime used GLiClass 0.1.20, Transformers 5.17.0,
tokenizers 0.23.2, huggingface-hub 1.31.0, the same PyTorch and NumPy versions,
and the approximately 151M-parameter
[`knowledgator/gliclass-modern-base-v3.0`](https://huggingface.co/knowledgator/gliclass-modern-base-v3.0/tree/ac369222ca4375ca66ebaf7fb5220f223514c035),
commit `ac369222ca4375ca66ebaf7fb5220f223514c035`, Apache-2.0.

**Use separate environments for the model and compare extras.** GLiNER2's local
and training extras require Transformers below 5; GLiClass 0.1.20 requires
Transformers 5 or newer. Installing them together is not a supported runtime.

## Latency measurement

The benchmark used one synthetic text about a late order and a courteous support
reply, with 1, 5 and 20 repeated binary politeness questions. Each case had three
warm-up calls and 20 timed calls. Device synchronization surrounded timing.
The GLiNER implementation evaluates all questions in one schema. GLiClass batches
one prompted text per question, repeating the text. Their token totals therefore
have different meanings; the comparison measures the same requested answers,
not equal-length tensors. These repeated questions do not establish accuracy on
diverse rubrics or independent production workloads.

| Model | Questions | Tokens including question schema | Warm p50 | Warm p95 |
| --- | ---: | ---: | ---: | ---: |
| GLiNER2.5 multilingual | 1 | 81 | 8.39 ms | 8.59 ms |
| GLiNER2.5 multilingual | 5 | 237 | 22.56 ms | 29.97 ms |
| GLiNER2.5 multilingual | 20 | 822 | 100.38 ms | 103.12 ms |
| GLiClass modern base v3 | 1 | 54 | 7.92 ms | 11.55 ms |
| GLiClass modern base v3 | 5 | 270 total over 5 texts | 15.45 ms | 16.47 ms |
| GLiClass modern base v3 | 20 | 1,080 total over 20 texts | 44.54 ms | 45.04 ms |

Cold initialization plus the first prediction took 4.75 seconds for GLiNER and
2.03 seconds for GLiClass. Warm figures include local input preparation and score
conversion; they exclude HTTP, service authorization, calibration, queueing and
human review. They are not service latency guarantees. A 20-sample p95 is a small
measurement, not a reliable tail-latency estimate for deployment sizing.

GLiNER process-lifetime peak RSS was 3.65 GB, with sampled MPS driver allocation
peaking at 2.32 GB. GLiClass measured 0.60 GB process RSS and 1.27 GB sampled MPS
driver allocation. Values use decimal GB. These quantities overlap on unified
memory: do not add them. Sampling every 10 ms can miss short-lived allocations.
Neither run estimates concurrency or training peak memory.

The exact synthetic benchmark input was:

> The customer asks when the delayed order will arrive. The reply says: Thank you for your patience. Please send your tracking number so we can investigate.

Each question used `Is this reply polite?`, with `yes: The reply is courteous`
and `no: The reply is rude`. Only the question ID changed between repetitions.
No model-generated explanations were requested.

## Training and offline verification

Periodic retraining starts from the original reviewed public GLiNER weights,
SHA-256 `c1ff4ec0bc00031c15530b8f3c33d3677f27949e6a0cb52e1247a6224b6c5395`,
using a new snapshot containing all currently authorized examples. The trainer
rejects warm-starting from any private or adapted checkpoint, including a renamed
checkpoint with its training metadata removed. This avoids inheriting private
weights whose ancestor grants may later be revoked. Recursive adapter ancestry
and warm-start training are not supported by this release.

Both full training and rank-8 LoRA completed a real FP32 optimizer step on MPS,
using in-memory authored synthetic fixtures from an encrypted learning store.
The tests used explicit AI-use and private-training grants. Training consumed
only the train partition; calibration and test examples remained excluded.

A fresh installation using only the documented model/test extras also passed
the complete LoRA command-line flow: train, reload, fit weight-bound calibration,
register a signed shadow bundle, score the independent test partition, reject
selective promotion of synthetic evidence, and retire the model after revocation.
Outbound sockets were blocked throughout this 25.35-second check. It establishes
the lifecycle mechanics, not a useful model trained from six synthetic examples.

| Check | Full training | LoRA |
| --- | --- | --- |
| Trainable parameters | 287,355,159 | 1,357,832 (0.47% of adapter-augmented model) |
| Real backward pass and optimizer step | Passed | Passed |
| Sampled trainable parameters changed | Passed | Passed |
| Save portable full checkpoint | Passed | Passed after adapter merge |
| Reload and reproduce probe scores within 0.0001 | Passed | Passed |
| Revoke originating grant and reject model use | Passed | Passed |

The initial separate smoke runs produced zero observed score difference after
reload. The reproducible two-mode pytest run completed in 19.51 seconds total.
That duration includes loading and file operations; it is not a training-speed
benchmark. One step does not demonstrate useful learning or rating accuracy.
Synthetic fixture labels exercise the software's independent-reference path;
they are not presented as real customer or independent human evaluation data.

Outbound socket connection attempts were replaced with errors during real
inference, training, checkpoint reload and benchmark processes. Only the explicit
provisioning step downloaded public model files. Application library offline
flags supplement this test; they do not replace a deployment network firewall.

To rerun the primary checks after explicit provisioning:

```sh
RATELOOP_TEST_MODEL_DIR=/absolute/path/to/provisioned/gliner \
RATELOOP_TEST_DEVICE=mps \
python -m pytest tests/test_backends.py::test_real_checkpoint_scores_without_network \
  tests/test_training_real.py
```

Use `RATELOOP_TEST_GLICLASS_DIR` with `tests/test_gliclass.py` in the separate
comparison environment. Without these explicit environment variables, expensive
hardware tests skip; ordinary contract/privacy tests remain runnable without
model dependencies or downloads.

## Compatibility decisions and remaining evidence

- GLiNER's upstream trainer chooses CPU or CUDA, so the evaluator makes MPS an
  explicit device override while retaining upstream loss, optimizer and saving.
  Mixed precision and fused optimizers are disabled for this verified path.
- GLiNER's serialized tokenizer metadata needs its upstream compatibility
  fallback. Protobuf is explicitly required so Transformers does not hide that
  known fallback behind an unrelated missing dependency error.
- DeBERTa under the verified Transformers 4 runtime falls back from SDPA to eager
  attention. These measurements include that fallback.
- GLiClass's current pipeline default marker differs from the v3 checkpoint's
  marker. The backend derives marker strings from existing checkpoint token IDs;
  it does not create new, untrained embeddings.
- A saved GLiNER checkpoint produced an upstream tokenizer regex warning.
  Prediction round trips passed. No tokenizer behavior was silently changed to
  suppress the warning; broader multilingual parity still needs representative
  tests before deployment.
- No production calibration, German rating-quality comparison, human-review
  reduction claim, long-running concurrency test, air-gapped installation audit,
  CUDA run or quantized model benchmark is established by this record.
  Production promotion still requires the independent held-out quality gates.
## Local HTTP service

The same M5 Max ran the native service with the pinned GLiNER model on MPS, the committed English reply example, one criterion, scoped authentication and encrypted persistence. Across 20 warm requests after two warm-ups, client-observed HTTP p50 was **17.25 ms** and p95 **19.75 ms**. Exact idempotent retry, cross-workspace rejection and metadata-only bundle registration export passed. All outputs correctly abstained as `uncalibrated`. This is a small synthetic, single-worker, small-store measurement; it does not establish customer accuracy, concurrent throughput or large-dataset latency.

## Alpha outbound worker acceptance, 17 September 2026

Worker implementation through `8303fa5` was exercised against the actual RateLoop Next.js job, grant, receipt and label-export route handlers backed by disposable PostgreSQL. The test adapter replaced only object storage and supplied authored public/synthetic cases; it did not replace GLiNER, authorization, job leasing, receipt validation, invited-review persistence or human-result projection. Native MPS ran the pinned multilingual checkpoint.

The test created a website job, retained its input under earlier private-learning consent, recovered its fenced claim after a lost/blinded receipt acknowledgment, and completed on attempt two. Before the fixture reviewer responses were submitted, the website job projection withheld the AI label and score. Automated test accounts submitted scripted responses through the real invited-panel routes, completing the review and releasing the AI label (`approved`, uncalibrated raw score about 0.776) alongside a positive outcome in the human-review projection. The connector imported one synthetic label through its independent-reference pathway after release with no quarantine or rejected items. These were automated test responses, not judgments collected from independent people. This tests the software path, not model accuracy or a live www deployment.

The English/German website template commitments matched the application SDK and job projections: `sha256:bb22546e25ff9d5587c60543653c111aeeac565d2e2ebff2f0228ee3a87bf20f` and `sha256:96b3f17aff30d79606537f0f08b556aad5f24966a90e40633a76c5c422bd6d18`.

Seventeen opted-in hardware checks passed in 35.25 seconds, including real inference, full and LoRA optimizer updates, save/reload, calibration and revocation. A further full/LoRA run with renewable execution leases and per-update authorization passed in 29.95 seconds. These model tests denied network connections. GPU access required running outside the process sandbox; a sandboxed process correctly reported MPS unavailable.

The acceptance driver then collected two further source-distinct website cases through the real invited-review persistence path. Three disjoint source groups produced one train, one calibration and one final-test group; the source-group split does not establish independence or representativeness of human raters. `scripts/alpha_e2e_operator.py` performed a real LoRA update, checkpoint reload, calibration and held-out scoring, and exported `gliner25-multi-alpha-candidate-v1` in shadow mode. After explicit owner registration and new model-scope permissions, a subsequent website job completed using that candidate bundle. Training and inference used the shared cross-process execution lock added in `d409dc7`.

The hosted browser journey is recorded separately below. The localhost route bridge itself does not establish hosted acceptance, reviewer-population evidence, an accuracy gate or permission to reduce human reviews.

## Hosted authorization clock regression, 17 September 2026

Before the clock fix, a live worker rejected its first authorization lease before claiming a job or running inference. Three subsequent authenticated diagnostic grant requests returned matching workspace, recipient and revocation-watermark values and exactly 900,000 ms lease durations. Their server issue timestamps were respectively 42 ms, 41 ms and 42 ms ahead of the Mac's local response-receipt time. This isolated the failure to the previous strict issue-time comparison; it was not a model, receipt or case-fencing failure.

Commit `a03f070bc80ca8749ebfd902b329dcfa1249acf0` permits at most five seconds of server-ahead issuance/revocation time for durable consent and issuance time for its execution lease. It conservatively ends local authorization at the earliest of server lease expiry minus five seconds, local receipt time plus 900 seconds, and finite consent expiry minus five seconds. The wire consent timestamps, hashes and stable lineage remain unchanged. Training collection still compares the original server consent and case timestamps without tolerance. Legacy immutable offline grants retain their original strict timestamp checks and expiry.

The complete non-hardware test run passed **181 tests**, with **four optional hardware tests skipped**. Regression cases cover the measured 42 ms difference, both five-second clock directions, rejection beyond the five-second future boundary, strictly positive lease duration up to 900 seconds, rejection at 900.001 seconds, exact and near expiry, finite consent expiry, prompt revocation, next-day renewal with unchanged consent identity, and the worker's unchanged 125-second acceptance boundary for a 120-second job lease. The published commit also passed [GitHub CI run 35237365027](https://github.com/Noc2/rateloop-evaluator/actions/runs/35237365027). No inference or training was rerun for this clock regression, and these checks do not establish resilience to host-clock rollback.

## Explicit model rollback regression, 17 September 2026

A hosted rehearsal completed four base-model cases, a real local training update and a subsequent candidate-model case. Its sixth case selected the base model on the website while the Mac still had the candidate active. The worker correctly refused this mismatch. The acceptance driver had omitted local rollback and ignored the worker's returned failure state; it waited for a prediction that could not complete. These cases used automated synthetic reviewer responses, not independent human judgments.

Commit `074d81701e45ca8a6b2485f9be2b76eb6e64541e` adds an explicit connected rollback command. It refreshes authorization under the shared model-execution lock and checks the exact expected previous bundle atomically before changing local activation. The separate website selection must match that restored local deployment. The worker's existing strict active-model check remains unchanged.

The complete non-hardware suite passed **189 tests**, with **four optional hardware tests skipped**, including the actual operator → CLI → signed-registry path. Regression cases verify successful base → candidate → explicit base rollback, refusal of wrong or foreign-language targets, revoked training lineage, unavailable authorization, concurrent model execution and a repeated rollback with no previous deployment. Refusals do not alter the current activation. The published commit passed [GitHub CI run 35245471485](https://github.com/Noc2/rateloop-evaluator/actions/runs/35245471485). No GPU operation or live-state mutation was used to verify this fix.

## Live hosted browser acceptance, 17 September 2026

Run `1adc8499cfaa4a7f` completed successfully at 16:29 UTC against `www.rateloop.ai`, serving application commit `6aefb50e4c4ac7c304957f7ddcadade7a988693e` from deployment `dpl_EpHqBTJaAu12AtFrSfXQ5gux7JGF`, with local evaluator commit `074d81701e45ca8a6b2485f9be2b76eb6e64541e`. Repeat capture `bef5548f70b646c6` passed all lanes on the same versions at approximately 16:44 UTC and retained private metadata and screenshots. Its invited-review journey took 10.2 minutes.

The complete journey passed ten smoke checks, four email-OTP sign-ins, two ordinary review panels and six AI cases: three English base-model cases, one German base-model case, one trained-candidate case and one English case after explicit local rollback. The Mac performed real MPS inference, one local optimizer step, calibration and held-out scoring. The browser observed worker training status, and the permission mirror renewed its execution lease while retaining consent identity. Paused evaluation was rejected with HTTP 409; revoked processing was rejected with HTTP 403. Explicit case erasure and disposable-workspace cleanup passed. The operator and connector credential files were verified absent after cleanup.

The six cases in the repeat capture produced these timings:

| Measured interval | Median | Range |
| --- | ---: | ---: |
| Local evaluation | 121.5 ms | 120–125 ms |
| Submission to observed receipt | 11.298 s | 11.110–11.703 s |
| Submission to scripted review completion | 18.465 s | 17.323–23.274 s |

Local evaluation is the receipt's `durationMs`: initial local permission/policy checks, tokenization, prediction and calibration. It excludes model loading and later authorization checks and persistence. The original capture labels this value `inferenceMs`; that field is not a pure neural-inference measurement. Submission-to-receipt time includes a fresh one-shot worker's startup, model loading, network traffic, postprocessing and observation polling. Its difference from local evaluation, originally named `queueAndTransportMs`, does not isolate queueing. Scripted review timing measures automated responses through the real review UI, not human response time. Six synthetic cases do not establish tail latency, sustained throughput or a service guarantee.

The retained English desktop screenshot at 1440 px and German mobile screenshot at 390 px were inspected. Neither showed horizontal page or panel overflow; the wide mobile comparison stayed within its inner horizontal scroller. The tall panel capture included a fixed-header overlay. This bounded check is not a complete accessibility or responsive-layout audit.

The exact application deployment was promoted at 16:29:36.807 UTC. Vercel Current, the tested hostname and enabled cron configuration matched it; actual scheduled maintenance returned HTTP 200 at 16:30:12.287 UTC and scheduled deliveries at 16:31:35.184 UTC.

All cases and reviewer responses were synthetic and scripted through the real application. All six AI outcomes remained `uncertain` in shadow mode. Raw predicted labels agreed with the scripted responses, but that agreement is not accuracy evidence. Passing this flow establishes integration and control mechanics, not independent human evidence or permission to reduce human review.


## Hosted CPU packaging acceptance — 2026-09-25

Runtime commits `27f1bc2` and `7f7e47f` added shared warm model startup, presence before owner enablement, private restart-preserving bootstrap, and metadata-only health. Python 3.12 local checks passed **262 tests**, with four opt-in real-model tests skipped; the interface contract checker and source/wheel build also passed.

The separate container acceptance used the actual pinned `fastino/gliner2.5-multi-v1` checkpoint at revision `235cf92d6d4318da9bfca0d08975c8fa7250d13b`, CPU PyTorch `2.14.0+cpu`, GLiNER `2.0.0`, Transformers `4.57.6` and the image's exact dependency constraints. Docker ran `linux/amd64` under emulation on an Apple ARM host, with **one CPU, 2,500,000,000 bytes RAM and no swap**, and networking disabled. The only remote behavior was a test-only paused-workspace transport mounted outside the image.

Cold startup reached healthy after 59 seconds; a restart reached healthy after 48 seconds. Both showed UID/GID 10001 and zero OOM kills. SIGTERM exited normally in 1.32 seconds; encrypted-state key identity survived restart. Both language registrations loaded the same model and health remained unavailable until warmup and a successful authenticated fixture poll/heartbeat. Cgroup memory reached its cap while loading/reclaiming model file cache, without an OOM; this is not evidence for a smaller memory allocation.

A separate synthetic model measurement using the same exact 2,500,000,000-byte limit measured 0.50–0.61 seconds for short English/German examples and 2.15–2.40 seconds at the exact 512-token template limit. Model loading took 11.1 seconds; maximum cgroup usage was 2,499,997,696 bytes with zero OOM events or kills. These timings include amd64 emulation, carry no accuracy claim and are not a Railway SLA. Repeat `scripts/measure_cpu.py` on the deployed hardware before making throughput claims. The container lifecycle check is reproducible with `scripts/test_hosted_container.py`; it does not prove live website authorization, persisted results, human blinding or a deployed unattended service. Those require separate live application acceptance.


The hosted image and native Mac constraint files subsequently moved to `cryptography==49.0.0`, with the package range restricted to `>=49,<50`. This is the fixed release identified by [PyCA advisory GHSA-jwv3-5hgf-82ww](https://github.com/pyca/cryptography/security/advisories/GHSA-jwv3-5hgf-82ww) for repeated self-signed certificate path-building. PyPI supplied non-yanked Python 3.12-compatible Linux AMD64 wheels. The full Python suite, contract check and dependency consistency check passed after the update; Fernet encryption and Ed25519 bundle signing use the unchanged interfaces exercised by those tests. The rebuilt patched image also repeated the real-model container acceptance at the exact RAM/CPU cap: cold/restart readiness in 60.0/47.4 seconds, SIGTERM exit 0 in 0.90 seconds, identical encryption identity and zero OOM kills.


The optional server-only registration handoff passed **264 tests** (four opt-in skips) and real-model container acceptance with `RATELOOP_HOSTED_EXPORT_REGISTRATIONS=1`. Both tagged registration JSON objects exactly matched the cloud-style volume exports, repeated unchanged after restart, and omitted the synthetic credential and workspace configuration. At the same exact cap, export-enabled cold/restart readiness took 78.2/64.2 seconds, graceful shutdown took 0.74 seconds, and OOM kills remained zero. Normal bootstrap emits no registration metadata. This explicit handoff preserves each actual installation's activation evidence; it does not make separately bootstrapped local and cloud activation timestamps identical.

## Custom dataset and candidate rehearsal — 26 September 2026

Evaluator `450b7f204f96ce52f76dc5389910f9cd2cd3b1b2` completed a real local rehearsal of the new dataset
CLI and candidate lifecycle, using the already provisioned GLiNER checkpoint and native MPS runtime described above.
The authored fixture asked whether a summary was supported by its supplied source; it was not a customer-reply task.
Twelve synthetic, explicitly labeled source groups were imported under **private-training permission only**, with
`input.text`, `input.evidence` and `imported_labels` scopes. No inference, shared-contribution or public-weight right
was inferred from import. The immutable snapshot contained eight train, one calibration and three final-test groups.

The baseline was compared before training. A single real LoRA optimizer step used FP32/MPS, `epochs=1`, `max_steps=1`, batch size 1,
learning rate `1e-5`, rank 8 and seed 42. The training command took 9.73 seconds including checkpoint saving/reload;
trainable parameters changed, and the merged checkpoint's prediction reload difference was **0.0** (required maximum
`1e-4`). The manifest retained the imported dataset version and `synthetic: 8` training-label provenance. No test or
calibration example entered training.

The baseline and candidate were then compared on the same frozen snapshot and the same three held-out source-group
representatives. Both matched all three synthetic reference labels; the 95% Wilson interval was **43.85%–100%**.
This tiny mechanics fixture establishes neither quality, improvement from training, representative accuracy nor
independent human agreement. The report correctly declared zero independent references and `quality_gate: false`.
An attempted independent calibration from these imported labels was rejected.

Registering the trained candidate and exporting its website registration preserved the active baseline. Separate,
explicit shadow activation selected the candidate; explicit rollback restored the expected baseline. All steps took
28.94 seconds combined, excluding the initial sandbox attempt that correctly reported MPS unavailable. Networking
was blocked by rejecting every outbound socket connection throughout the successful run; no models were downloaded,
no external service was called, and no paid infrastructure or hosted state changed. Raw fixtures, keys, checkpoint and
machine-readable reports remained outside Git. This proves local import → snapshot → train → reload → compare →
register/export → explicit activation/rollback mechanics. Website import/training UI, hosted queue dispatch and a
website result from this custom candidate require their separate end-to-end acceptance.

## Optional training runner and bounded checkpoint cache — 26 September 2026

Evaluator `041ac94b017aa206ecb913d195506be4010742a6` passed **369 tests**, with four optional hardware skips, and the portable contract checker. Authenticated-transport tests cover dataset import, frozen comparison, candidate export, separate activation and rollback, cancellation during optimization, revoked or expired dataset erasure, exact completion acknowledgments, and restart reconciliation. Neural computation is stubbed in those transport tests; the separate real optimizer rehearsal above establishes checkpoint mechanics.

A durable isolated operator runtime at that source version reused the existing pinned public checkpoint and installed dependencies. Its two English/German registrations shared one model instance. Real native MPS inference accepted custom questions about whether a summary mentions a budget; no dataset, grant, human label or quality claim was created. A separate cache check loaded the prior synthetic trained checkpoint, then the public checkpoint, then the trained checkpoint again. Each completed load reported 1,158,335,488 allocated MPS bytes; closing the cache returned allocated MPS memory to zero. This is bounded resident-checkpoint evidence, not a process-RAM peak measurement or a service throughput guarantee. Unit checks additionally verify eviction before loading a new private checkpoint, exact queued bundle routing, and rejection of revoked bundles before loading.

The prepared operator uses its own private state, signing key and workspace credential. Its outbound service definition was prepared but not loaded during this check. No model download, hosted service change, website dataset request, private-weight upload or additional training occurred. The always-on hosted CPU deployment and website training journey still require their separate live acceptance; the local checks do not establish those outcomes.

## Bilingual diagnostics and guiding examples — 27 September 2026

The diagnostic suite and runtime content published in
`24486a6013679a8b2f7fdfde309d7105bfb2869a` were exercised with the local runtime above.
`diagnostic_cases()` fixes 24 authored cases: four per task and language, for numeric-budget
criteria, supplied-evidence support and email-address presence in English and German.
The suite commitment is
`sha256:92656644bab0cb02ad80eb72bbfe639e1dd5c815ebfa5c44908efdb58875b2c4`.
These are synthetic regression labels with zero independent references. Each report declares
`quality_gate: false` and `activation_changed: false`; these results do not measure representative accuracy.

Two Apache-2.0 checkpoints were explicitly provisioned and compared:

| Checkpoint | Pinned upstream revision | `model.safetensors` SHA-256 |
| --- | --- | --- |
| Current `fastino/gliner2.5-multi-v1` | `235cf92d6d4318da9bfca0d08975c8fa7250d13b` | `c1ff4ec0bc00031c15530b8f3c33d3677f27949e6a0cb52e1247a6224b6c5395` |
| Challenger `fastino/GLiNER2.5-multi-Decide` | `6bc1d43d201b0691e733626389af8c57eea3ea68` | `9efe0f88c99f2aa794452e9559dc60e98d60d9fa2bf1b60cf2710411b6da5b4e` |

Both used GLiNER2 2.0.0, the unchanged `Schema.classification` interface with
`multi_label=True`, `class_act="softmax"`, `cls_threshold=0.0`, complete returned label
scores and adapter version 1. The multi-label decoder setting exposes every score;
explicit softmax retains mutually exclusive binary semantics. No scores were calibrated.
Prediction used the production context/evidence/artifact rendering and the existing
`approved`/`rejected` IDs with `Yes`/`No` or `Ja`/`Nein` descriptions.

Each checkpoint ran in its own fresh process, sequentially on native Mac CPU with one
PyTorch intra-op and one inter-op thread, FP32. The process blocked outbound socket
connections; only explicit prior provisioning downloaded model files. The 24 cases used
66–109 tokens including the complete schema, below the 512-token task limit. Timing
covered one prediction per distinct case after token counting, with no repeated warmups.
It excludes loading, network, authorization, queues and human review.

| Configuration | Correct / 24 | Balanced agreement | Prediction p50 / p95 | Sampled peak RSS / end RSS |
| --- | ---: | ---: | ---: | ---: |
| Current base | 11 | 50.71% | 36.23 / 43.09 ms | 3.628 / 2.518 GB |
| Decide challenger | 10 | 47.14% | 38.68 / 46.52 ms | 3.577 / 2.517 GB |
| Current base with two guiding examples per task/language | 11 | 49.29% | 46.09 / 66.48 ms | 3.518 / 2.525 GB |

Balanced agreement is mean recall over the two reference labels; the suite contains
10 positive and 14 negative references. A constant negative label would match 14/24,
which further limits any interpretation of the raw agreement count. Model loading took
4.76, 5.04 and 4.87 seconds respectively. RSS uses decimal GB and includes initialization;
the 10 ms sampler may miss brief allocations. This Mac workload neither sizes the hosted
Linux service nor proves that a challenger fits its 2,500,000,000-byte cap. No new hosted
service, paid API, cloud trainer or additional monthly allocation was created.

| Slice (four cases each) | Base | Decide | Guided base |
| --- | ---: | ---: | ---: |
| Numeric budget / English | 4 | 3 | 4 |
| Numeric budget / German | 3 | 3 | 3 |
| Evidence support / English | 1 | 1 | 1 |
| Evidence support / German | 1 | 1 | 1 |
| Email presence / English | 1 | 1 | 1 |
| Email presence / German | 1 | 1 | 1 |

All three configurations falsely approved contradictory, missing and partially supported
evidence cases in both languages. The base falsely approved email placeholders and URLs,
and rejected an actual email address when the surrounding sentence denied its presence.
Those three email failures occurred in both languages. German budget evaluation confused
an unrelated number with a budget; Decide also failed the English explicit no-budget case.
These include large-margin errors: the base rejected the English negated-claim email case
with raw score approximately 0.999987 and approved the unsupported partial-evidence case
at approximately 0.992009. Raw score spread is therefore not a correctness probability.

The guidance experiment supplied two authored examples per task/language, distinct from
all test texts, without optimization or a training run. Its full suite commitment is
`sha256:7e07b3a6210b3b088556f3ddc68b9cb329dcd0d739054d0ad80b2340028066db`;
combined lengths were 97–198 tokens. Guidance corrected the English email placeholder but
introduced an error on an ordinary email address, leaving total agreement unchanged.
The source examples and machine-readable reports were retained outside Git.

Two exploratory adapter experiments followed the fixed comparison. Replacing only internal
label IDs with `yes`/`no` produced 12/24, fixing the German budget case while leaving every
evidence/email slice at 1/4. Replacing short answer descriptions with explicit semantic
descriptions produced 13/24, with evidence slices at 2/4 but English email presence at 0/4.
These post-baseline design probes did not establish a universally inverted parser or a
reliable fix. They were not production adapter changes, independent benchmarks, new
calibration or grounds for promotion. The original checkpoint and score interface remain
the hosted default; Decide remains an unpromoted local challenger.

### Demonstration contract and training checks

Runtime changes `6bd8517` and `5f68e3a` preserve old template commitments when examples
are absent/empty and bind nonempty examples to the exact question commitment. The SDK
golden fixture matches Python at
`sha256:3647abb556e0a4938e06a74206a76adc54bee3867504d190b1a8fd7ec5ae2c56`.
Tests cover label membership, strict fields, four-example/600-unit/1,600-unit bounds,
UTF-16 emoji boundaries, control characters, complete schema token counting, denied
cross-template permissions and refusal to reuse calibration from an earlier rubric.
Import, training and comparison reject disclosed example text reused as dataset evidence
after the same ECMAScript whitespace normalization; semantic near-duplicates still require
source-group review. GLiClass fails explicitly on examples instead of dropping them.

A real CPU test counted and evaluated examples with network connections blocked, preserving
the complete two-label softmax vector. The opted-in LoRA test in `8512b19` then used six
authored synthetic text fixtures and two separate guiding examples on native MPS, rank 8,
`epochs=1`, `max_steps=1`, one optimizer step, batch size 1, learning rate `1e-5`, seed 42 and FP32.
It passed in **21.13 seconds**, including optimizer execution, sampled parameter change,
merged checkpoint save/reload within `1e-4`, weight-bound calibration, refusal to promote
synthetic evidence and model retirement after grant revocation. The train partition alone
entered the optimizer. Socket connections remained blocked throughout. The initial sandbox
attempt correctly reported MPS unavailable; the explicit local GPU run succeeded outside
that sandbox. This establishes demonstration-aware lifecycle mechanics, not beneficial
learning, independent human evidence or a connected website training run.


## Connected-training export regression — 28 September 2026

The authorized website rehearsal against runtime
`e9a7980ad0b5d91e5a6c524d78e5f4b925994f35` completed all 20 optimizer steps and
saved a 5.2 MB LoRA adapter, but the website rejected the candidate receipt. The
upstream checkpoint export reserialized `tokenizer.json` and
`tokenizer_config.json`, changing the registered tokenizer commitment from the
permitted base. The strict server correctly rejected that identity mismatch.
The worker then sent an unsupported `retryable` field in its training failure
report, which was also rejected, leaving the website job appearing to run while
the worker had returned to polling. Encrypted operation history and local
artifacts survived; no candidate was activated.

Runtime `e0d04d4` fixes the failure contract and preserves failure intent until a
matching acknowledgement or an explicit lost-lease response. Its 28 training-worker
tests cover strict fields, rejected/unavailable/ambiguous failure acknowledgements,
404/409/410 fencing, durable completion, cancellation, revocation and model-switch
reconciliation. Runtime `98047ff` preserves exact source tokenizer assets, removes
new export sidecars absent from that source, and checks the loaded fast tokenizer
semantics before training, after training and after reload. Training and registration
share the existing tokenizer commitment function; the wire hash has not changed.

Verification on `98047ff`:

- 467 tests passed and five optional model/hardware tests were skipped. One localhost
  health-server test initially lacked sandbox binding permission; that same test
  passed with local binding enabled, without a code change.
- The focused training, worker, CLI and registry set passed 57 tests.
- The explicit real MPS LoRA test passed in 31.97 seconds with the already provisioned
  pinned base, six authored synthetic cases and two separate guiding examples, one
  optimizer step, rank 8, batch size 1, FP32, seed 42 and learning rate `1e-5`.
  Socket connections were denied throughout. It verified changed trainable parameters,
  full tokenizer semantics, exact source/checkpoint/export tokenizer commitments,
  prediction reload error below `1e-4`, weight-bound calibration, synthetic promotion
  rejection and grant revocation. Only the train split entered the optimizer.

These checks establish the corrected training/export mechanics, not improved model
quality or completion of the subsequent website rehearsal. Re-run that journey on
the updated operator runtime and a new candidate ID; preserve the rejected signed
candidate's immutable identity. The hosted base, score interface, service resources
and paid infrastructure allocation are unchanged.


The final retry guard `189201e435815554eb1dc9f425a81d93cd9342e9` also preserves a
failure intent when the server later reclaims that operation under a new lease.
All 468 tests passed with five optional skips in a single final run (8.06 seconds),
including 29 training-worker tests. No neural-training code changed after the real
MPS check above.

The website recipe used in this rehearsal was bounded by at most 20 optimizer steps; the independent
operator capability ceiling is 200. Upstream GLiNER derives the number of data
passes from a positive `max_steps`, overriding `num_epochs`. The 20-step connected
attempt traversed its small train partition across three passes. Historical
one-step checks above record the requested epoch setting, not a guarantee that
every training row was consumed exactly once. This detail does not expand the
step budget, data permission or held-out partitions.

## Public quality and validation gates — 28 September 2026

**Interpretation update:** these experiments preceded the immutable-schema
preprocessing correction. The pinned upstream trainer could alias labels and
then reinsert the original correct label with a negative target, and could drop
the rubric's descriptions/examples. Preserve the measurements below as evidence
of the affected implementation; they cannot distinguish inherent model limits
from this preprocessing defect. The validation rejection remains valid for the
actual candidates tested. Corrected-code regression runs are separate evidence,
with unchanged frozen datasets, recipe and gates; no historical result is
silently replaced or upgraded into a quality claim.

Training implementation `4f9baa7` was exercised on the existing pinned public
GLiNER2.5 model, offline on the Mac's MPS device. Source: NVIDIA HelpSteer2,
revision `990b2711a36180dd19d9c94b8627844866f8982a`, CC-BY-4.0,
[official attribution and data card](https://huggingface.co/datasets/nvidia/HelpSteer2).
The `public_quality.py` source manifest pins both compressed source files by SHA256.
No model was downloaded, private data uploaded, paid service created, website
candidate registered or active evaluator replaced during these experiments.

Each English criterion used 400 development prompt groups. Human scores 0–1 and
3–4 were mapped to the negative and positive labels; score 2 was excluded. Exact
token counting excluded oversized inputs without truncation. One response per
normalized source prompt counted as one observation; duplicate answers crossing
development/final partitions were excluded. These are deliberately bounded,
short-input, clear-label public cohorts, not a representative sample of every
RateLoop task. Their external human labels do not become authenticated RateLoop
blind judgments. Unknown contamination of the public base's pretraining is an
additional limitation.

The initial diagnostic prototype reserved upstream validation as final test and
used the natural label mixture with a maximum of 200 optimizer steps. It exposed
an accuracy illusion that led to the final balanced recipe and baseline gate:

| Criterion / 199 final prompt groups | Base agreement | Candidate agreement | Base balanced agreement | Candidate balanced agreement | False approvals, base → candidate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Helpfulness | 72.9% | 77.4% | 61.2% | 52.4% | 27 → 42 |
| Correctness | 73.9% | 79.9% | 63.1% | 51.9% | 22 → 38 |

Both diagnostic optimizers stopped at step 125 and selected step 50 by development
validation alone; the final results were not used for checkpoint selection.
Their merged weights saved and reloaded correctly, but neither candidate was
recommended. This prototype diagnosis is not an acceptance claim for the final
recipe. Diagnostic candidate weight hashes were
`93aba453a457cd25ba1eb50d6c51c7e1a817944dd13b01e1aa5ae225b45981d4`
(helpfulness) and
`e5b54f756924a34a39a00838c4e43b7845440efc994b2863add099e4dc6f4985`
(correctness).

Before testing the final balanced recipe, a fresh final set of 200 groups per
criterion was frozen from the unused high-hash tail of upstream training prompts.
Those prompts were excluded from development; both original 400-group development
commitments remained identical. No prior final group was reused. The final
recipe used rank-8 LoRA, FP32, seed 42, batch size 2, learning rate `1e-4`, at most
200 updates, validation every 25 updates and patience 3. Fixed group-hash
validation contained 63 groups for helpfulness (51 positive / 12 negative) and
65 for correctness (55 / 10); 40 different groups per task were reserved for
calibration. Label balancing resampled only optimizer source-group representatives.

The helpfulness run stopped at 125 updates; correctness stopped at 150. Neither
produced a checkpoint with greater validation balanced agreement and no per-class
recall regression against the baseline. Both returned `ValidationQualityError`
and wrote a content-free selection report. No model artifact or model lineage was
published, and candidate final-test scoring was skipped. This is the intended
rejection path, not a training-infrastructure failure or evidence of improvement.
The new final cohort remains unused for selecting a later candidate.

Separate temperature fits on the 40 calibration groups improved the base's score
calibration on the new final cohorts, without changing its predicted labels:

| Criterion / 200 final groups | Balanced agreement | Raw → fitted Brier score | Raw → fitted ECE (10 bins) | Fitted temperature |
| --- | ---: | ---: | ---: | ---: |
| Helpfulness | 60.34% | 0.4037 → 0.3544 | 0.1713 → 0.0751 | 2.8500 |
| Correctness | 60.42% | 0.4001 → 0.3683 | 0.1545 → 0.0836 | 5.0751 |

Brier uses the summed squared error across both labels. ECE is descriptive and
depends on the selected bins and sample. These fitted values are task-specific
public-label evidence; they were not imported into customer workspaces and do
not authorize displaying a universal correctness probability. Forty calibration
groups remain a small pilot. The base's modest discrimination supports retaining
human review or task-specific deterministic/reference checks for consequential
decisions; it does not support a general accuracy claim.

The public benchmark and CLI/training/quality tests passed 40 focused tests.
Private local reports contain source/filter counts, frozen cohort commitments,
confusion matrices, class recall, calibration bins, manifests and the selection
history. Their SHA256 commitments are:

- Initial helpfulness: `ea3064fa4a2c8d768250130edfa76c902c7d3529b909ee05102fa6461f4b8e7f`.
- Initial correctness: `ebbf68988aefe79465667d950e8bac60f908340f9babaa51574757b41e58c1b0`.
- Final helpfulness rejection: `a784ddb4c789722f35eebda2bc3f609ae296c6772b0498baad067fb0d47ab808`.
- Final correctness rejection: `b760effdb4ca8f82bf5dd7e3d49f051560354f76ba452786ed06b457f43af0e1`.

## Hosted private-model CPU capacity — 28 September 2026

The isolated hosted runtime and one-resident-checkpoint cache were checked with
an existing pinned `linux/amd64` CPU image, the current evaluator source mounted
read-only, networking disabled, one CPU, an 8 GiB memory ceiling and
`--memory-swap 8g` (zero permitted swap). This is an emulated amd64 Docker run on
Apple hardware, not a Railway latency SLA or a customer-quality benchmark.

The sizing script used 160 authored synthetic examples, each verified at exactly
512 tokens including the same rubric/context delimiters as live inference. Its
explicit fixed-budget capacity recipe ran 25 LoRA optimizer steps, saved and
reloaded the merged checkpoint, and compared the base/candidate on 24 frozen
source groups. Training plus comparison took 294.89 seconds. Container memory
peaked at 7,688,486,912 bytes (7.16 GiB), with no OOM events or swap. Subsequent
candidate/base/candidate switches took 9.91, 10.08 and 9.90 seconds. Three minutes
of idle observation showed that file cache could retain both checkpoints even
though only one tensor model was resident: total cgroup memory stayed at about
4.494 GB, of which 2.394 GB was file cache; working set was about 3.825 GB.

An earlier preliminary 5 GiB run used approximately 1 GB of swap and therefore
does **not** establish that 5 GiB is enough. The 8 GiB no-swap measurement is the
capacity evidence. Neither synthetic run is an improvement claim: both models
matched every label in the small authored capacity holdout. The production
validation-selected recipe is separately tested and still rejects candidates
that do not improve validation without class regression.

Runtime `c086afa` adds a read-only `POSIX_FADV_DONTNEED` hint for only the known
checkpoint file after load/unload. It does not delete files, alter model bytes,
flush global caches, or change predictions. The first before/after experiment
reduced cgroup memory from 4.720 GB to 2.421 GB and working set from 4.563 GB to
2.264 GB; candidate predictions remained exactly equal after the hint. A second
run exercised the committed automatic load/unload path through four actual
base/private switches: total memory was already about 2.068 GB and working set
2.036 GB before any additional manual hint; repeating the hint changed neither.
Both runs had zero swap. The final private checkpoint was produced by the
separate authorized synthetic budget experiment, not copied from a customer's
private model.

`scripts/measure_hosted_training.py` is the reproducible capacity harness. It
uses explicitly synthetic provenance, private temporary storage, a local pinned
public checkpoint, a fixed-budget training recipe, exact checkpoint switching,
cgroup/working-set/swap counters and three minutes of post-training idle
observation. It never authorizes a website dataset or activates a model. Public
and private inference remain serialized; training/comparison pause inference.
The hosted service additionally enforces a durable aggregate one-hour
training/comparison allowance per UTC day, a per-operation ceiling and at least
3 GiB of free disk before training; activation, rollback, permission revocation
and already-computed result reconciliation are not blocked by that compute
budget. The cap and measurements support a small pilot cost estimate, not an
unlimited-throughput or fixed-price promise; Railway's live measurements and
retained volume usage must be checked after deployment.
