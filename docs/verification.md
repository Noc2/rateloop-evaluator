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
