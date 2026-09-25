# Local model verification

This maintained record describes the local implementation checks performed on
2026-09-17 for `rateloop-evaluator` 0.1.0. It verifies execution, not suitability
for replacing human ratings. No customer content was used.

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
