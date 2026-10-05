# RateLoop Evaluator

An Apache-2.0 local evaluator for explicit rating questions, built around GLiNER2.5 multilingual.
It runs independently of a RateLoop account. Customer examples and private adaptations are never public by default.

It provides authenticated local inference, full and LoRA fine-tuning, encrypted human-feedback storage, separate training permissions, grouped datasets, calibration, signed model bundles, rollback and a RateLoop connector. English and German templates are supported. GLiClass v3 is available as a benchmark challenger in a separate environment.

**Choose human review, AI ratings, or both in RateLoop.** AI-only cases return an advisory rating without opening a human review. When both are selected, the AI answer stays hidden until the human answer is frozen. Choosing AI does not qualify the model for automatic approval: uncertain results remain uncertain, and pretrained scores are not calibrated probabilities. A local selective mode exists behind independent held-out quality gates; the included synthetic examples cannot qualify it.

## Run locally

Use Python 3.12 and Node 24 for development. macOS uses native PyTorch MPS; CPU and CUDA are explicit alternatives. Provisioning downloads public weights once. Inference and training load local files without a cloud fallback.

```sh
git clone https://github.com/Noc2/rateloop-evaluator.git
cd rateloop-evaluator
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[model,test]'
.venv/bin/rateloop-evaluator init --workspace example-workspace
.venv/bin/rateloop-evaluator provision --model-dir "$HOME/rateloop-models/gliner25"
.venv/bin/rateloop-evaluator grant --right ai_use --template customer-reply-tone --hours 24 --evidence 'Owner enables local evaluation of this template'
.venv/bin/rateloop-evaluator register --activate --model-dir "$HOME/rateloop-models/gliner25" --request examples/reply-request.json
.venv/bin/rateloop-evaluator serve --bundle-id gliner25-multi-shadow-v1 --device mps
```

Replace `mps` with `cpu` on a machine without Apple GPU support. Use your actual workspace ID in both initialization and requests when connecting RateLoop. `init` writes a private local client credential to `~/.local/share/rateloop-evaluator/client.json`; do not paste it into logs or browser code. No training permission is created by initialization or AI opt-in.

From a second terminal, run a synthetic request without printing the token:

```sh
.venv/bin/python - <<'PY'
import json
from pathlib import Path
import httpx
credentials = json.loads((Path.home()/'.local/share/rateloop-evaluator/client.json').read_text())
request = json.loads(Path('examples/reply-request.json').read_text())
with httpx.Client(trust_env=False) as client:
    response = client.post('http://127.0.0.1:8765/v1/evaluate', json=request,
        headers={'Authorization':'Bearer '+credentials['token']})
    response.raise_for_status()
    print(response.json())
PY
```

The initial result contains typed labels and raw scores, with `outcome: uncertain` and `abstainReason: uncalibrated`. Template changes require a new registration and calibration scope. Missing scope, excessive length and unmet deadlines produce abstention; required evidence is never silently truncated. A request carries a stable case ID and optional conversation/source group ID, so related examples can stay in the same training split. Reuse its idempotency key only for the same case and input.

## Custom text questions without training

The pinned public GLiNER model can answer custom binary text questions in English or German without a dataset or training run. Build a canonical template with `custom_text_evaluation(language, prompt, positive_label, negative_label)`: one question (up to 500 UTF-16 code units), two distinct answer descriptions (up to 40 each), and a 512-token combined input budget. Input text, optional context, and evidence remain separate fields. The helper normalizes wording; the wire contract requires that exact normalized template.

Optionally supply up to four labeled guiding examples through `examples=[{"text": "...", "labelId": "approved"}]`. They become part of the immutable question and its token budget; they do not train the model. See the [bounded-example workflow](docs/operations.md#bounded-guiding-examples) before reusing calibration or importing a dataset.

Register a **new** public-base bundle using `register --custom-text --activate --model-dir <local-model> --request <seed-request>`. Its signed `task_capability` and exported `taskCapability` use `rateloop.evaluator.custom-binary-text.v1`. Exported metadata includes the complete template. Every task still requires consent for its exact template commitment and model identity; a capability alone grants no processing or training rights. Changing the question or labels changes that commitment. The worker returns an advisory label and raw scores with `uncertain` / `uncalibrated`; no probability of correctness or automatic approval is asserted.

Hosted configurations may add new capability bundles to the existing volume while preserving all existing bundle entries, workspace, worker, model path and agent identity. Add `"taskCapability":{"schemaVersion":"rateloop.evaluator.custom-binary-text.v1"}` to each new English/German bundle entry. Old registrations stay immutable and queued jobs keep their original identity. All registrations reuse the same verified local weights. Training-derived bundles stay bound to their exact rubric and never receive this general base-model capability.

## Connect your own Mac

Install from this public repository using the local setup above, then provision the pinned rating model once. In RateLoop, create a device pairing code. Use a fresh private state directory for each device/workspace connection:

```sh
rateloop-evaluator --state-dir "$HOME/.local/share/rateloop-office" connect \
  --model-dir "$HOME/rateloop-models/gliner25" --device mps
rateloop-evaluator --state-dir "$HOME/.local/share/rateloop-office" start
```

The connection command privately prompts for the short-lived code; never include it in shell arguments. `--token-stdin` is available for an explicit private pipe. The command registers the exact existing English/German approval and custom-text models and stores its scoped device credential with owner-only permissions. It grants no processing or training rights. Select this device for AI ratings in RateLoop after the worker starts. The worker opens no inbound port.

For operation while you are logged in to your Mac, replace `start` with `install --load`. Keep the model, state directory and Python environment in place. This is a login service: a sleeping or powered-off Mac cannot accept work. Use `start --once` for a one-poll diagnostic. CPU and CUDA are explicit alternatives to `--device mps`.

A missing response during the one-use claim has an unknown outcome. Revoke that device in RateLoop, create another pairing code, and use a new private state directory. The connector never silently retries the consumed code or prints the device credential.

## Integrate RateLoop

`register` creates a candidate by default. Training, registering, and exporting metadata do not switch the active model. Use `promote --mode shadow` after reviewing the candidate, or deliberately choose `register --activate` for the initial base model. Promotion changes the default; already queued jobs continue using their original activated bundle only while it remains configured and its exact consent is current. Revocation and expired evidence still block those jobs. Never-activated candidates cannot serve inference.

Export the registered bundle's metadata:

```sh
.venv/bin/rateloop-evaluator export-registration --bundle-id gliner25-multi-shadow-v1 --request examples/reply-request.json --output /private/path/bundle-registration.json
```

In the updated RateLoop workspace's **Agents → Results** tab, open **Set up AI** (or **AI settings**), import this file and enable the evaluator. Use a workspace API key with evaluation and telemetry scopes. For website jobs, use `examples/approval-request-en.json` or `examples/approval-request-de.json`, containing the same overall approval question as the human review; replace placeholder workspace and bundle IDs before registration.

The registration export now includes `scoreCapability` version `rateloop.evaluator-score-capability.v1` for
`rateloop-evaluator/gliner2`, adapter version 1, with `mutually_exclusive_softmax` semantics. This identifies the
adapter's complete raw class-score vector; it does not certify calibration or correctness. A compatible RateLoop
application can display its normalized entropy as experimental score spread without another inference call. Signed
receipts and result commitments are unchanged.

Upgrade the application before importing this metadata. Existing server bundle registrations are immutable and keep
their original capabilities: exporting new metadata does not upgrade an already registered bundle. For an unchanged
public base model, register and select a new bundle identity with the same intended question and weights, then export
and import that registration. Keep old bundles available for already queued jobs. For privately trained bundles,
preserve the training lineage and consent scope through the normal managed bundle workflow; do not rename or edit
a trained manifest to bypass identity checks. Historical results without the capability remain unmeasured. The
website rollout is controlled separately by its server-side workspace pilot flag.

Select the review option before submitting a case:

| Choice | Result |
| --- | --- |
| Human review | Human review; no AI job or AI-processing permission required. |
| Rating AI | AI rating as soon as the configured worker completes; no human review or human training label. |
| AI + human | Both ratings, with the AI answer withheld until the independent human answer freezes. |

AI choices require an authorized, registered model and a running worker. AI-only cases do not retain inputs for training, even when workspace learning is enabled. Use AI + human with separate learning consent to collect independent labels for improvement. Existing required human reviews remain in force.

The outbound `worker` command pulls authorized website cases over HTTPS, runs the pinned local model, posts fenced advisory results and synchronizes independently frozen human judgments. No public Mac port is opened. The Alpha website sends submitted cases to its configured RateLoop-operated worker; this is not a fully offline website. An always-on Linux CPU worker uses the same outbound protocol and shared English/German weights. [Hosted setup and readiness checks](docs/operations.md#always-on-hosted-cpu-worker). Results and provenance return as metadata; private training examples and weights stay local. [Install and run the worker](docs/operations.md#outbound-website-worker).

AI processing and private learning are separate durable owner permissions with exact credential, model and template scopes. Execution leases last at most 15 minutes and renew while connected. Enabling learning later does not retain previously queued inputs. Withdrawal retires affected models; case-erasure tombstones delete their retained examples. Legacy grants keep their original finite expiry.

The [connector guide](docs/connector.md) explains configuration, durable receipt delivery, blind audit selection and importing human outcomes. The [TypeScript client](clients/typescript/index.ts) calls your local service directly from a trusted server or agent. The [versioned interface](contracts/INTERFACE.md) and JSON schemas define cross-language commitments. The RateLoop SDK carries the same result contract.

Local AI inference does not make a cloud-connected RateLoop installation fully on-premises. Existing human-review submissions remain separately authorized and follow RateLoop's content policy. A complete customer-hosted human-review platform is outside this evaluator repository.

## Learn from opted-in human ratings

Grant `private_training` separately before retaining any examples. Issue separate, identity-bound reviewer credentials with `issue-reviewer-token`; inference agents cannot submit human feedback. Feedback must bind the exact input and template commitments. AI-exposed, duplicated, conflicting or non-independent labels are quarantined. An overall human verdict cannot silently become several criterion labels.

The [learning guide](docs/learning.md) covers the sequence: collect authorized independent labels → create a grouped snapshot → train from the public base → calibrate on independent groups → register an immutable candidate and operating policy → score the final test groups → promote only if its measured gate passes. Retrain from the public base using currently authorized data for each release; training on an already adapted private checkpoint is deliberately rejected until recursive parent lineage is implemented.

`ai_use`, `private_training`, `shared_contribution` and `public_weight_distribution` are independent rights. Private training does not authorize shared training or public weights. Public weights do not require publishing raw examples. Revocation retires affected managed datasets and models; it does not promise instant unlearning from copied weights.

## Public datasets and local diagnostics

Prepare a **local, explicitly obtained training-split JSONL export** with `prepare-public-dataset` before importing it in RateLoop. Supported adapters cover [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2), [HelpSteer3 Principle](https://huggingface.co/datasets/nvidia/HelpSteer3#principle), and [OpenPII 1M](https://huggingface.co/datasets/ai4privacy/pii-masking-openpii-1m). Their published dataset licenses are CC-BY-4.0; retain attribution and review the selected material. Other similarly named PII datasets can have different terms.

```sh
rateloop-evaluator prepare-public-dataset \
  --dataset helpsteer2 --file /private/path/helpsteer2-sample.jsonl \
  --revision <full-source-commit-sha> --source-sha256 <local-file-sha256> \
  --source-split train --language en --score correctness \
  --positive-min 3 --negative-max 1 --output-dir /private/path/prepared
```

The new private directory contains `examples.jsonl`, `template.json`, `manifest.json` and `excluded.jsonl`. Use the generated question and labels when importing `examples.jsonl` under **AI settings → Improve AI**. Keep the manifest and exclusions with the experiment. Select the manifest's `import_provenance` (`owner`, `ai_assisted`, or `synthetic`); external human annotations remain imported labels, not independent blind RateLoop reviews. The command does not download data, grant permissions, train, or activate a model.

- **HelpSteer2:** choose helpfulness, correctness or coherence and explicit score thresholds. Intermediate scores are preserved in exclusions for separate review, never silently relabeled.
- **HelpSteer3 Principle:** use `--dataset helpsteer3-principle --principle '<exact source principle>'` instead of score options. One criterion and one language per export; labels are AI-assisted.
- **OpenPII 1M:** use `--dataset openpii1m --entity EMAIL` instead of score options; English and German are supported. Positives and negatives follow the selected entity's validated annotations. No artificial clean negatives are generated; absence of one entity is not a privacy or legal-compliance verdict.

The converter accepts at most 64 MiB / 50,000 input rows and emits at most 2 MiB / 2,000 rows, selecting complete source groups deterministically. It refuses upstream validation/test splits so they cannot silently enter training. The existing snapshot flow freezes train/calibration/test assignments, keeping shared prompts, masked templates and formatting duplicates together. Review semantic relatives that source identifiers cannot detect. Exact model token limits still apply; no content is shortened automatically. File hashes identify the export; they do not prove who published it.

Run the small bilingual regression suite against an already provisioned local model:

```sh
rateloop-evaluator diagnose --model-dir /absolute/local/model --device mps \
  --output /private/path/diagnostics.json
```

Its 24 synthetic cases cover criterion compliance, evidence support and email presence, including negation and missing evidence. Reports include per-task/language errors, per-label recall, balanced agreement and prediction latency. These are execution/regression measurements, not representative accuracy, calibrated confidence or deployment qualification. Compare candidates on authorized, frozen held-out data with `compare`; imported labels cannot satisfy independent-human qualification gates. Training and activation remain separate explicit actions.

The [27 September comparison](docs/verification.md#bilingual-diagnostics-and-guiding-examples--27-september-2026) found 11/24 correct labels for the current base, 10/24 for the pinned Decide challenger, and 11/24 for the base with two guiding examples. Groundedness and email-presence cases exposed substantial errors, including confident errors. The hosted model remains unchanged; guidance and a newer checkpoint are not demonstrated general improvements. Decide is available only for [explicit local comparison](docs/operations.md#compare-the-pinned-decide-challenger).

## Hardware and verification

On an M5 Max with 128 GB unified memory, actual GLiNER inference, full fine-tuning, LoRA training, save/reload, and offline execution passed. A synthetic one-question local HTTP check measured about **20 ms warm p95** over 20 samples. This is not an accuracy result or service SLA. Model-only GLiNER warm p95 was roughly 9/30/103 ms for 1/5/20 short repeated questions. GLiClass's corresponding measurements were 12/16/45 ms. See [exact revisions, software versions and limits](docs/verification.md).

The compact model does not need 128 GB for inference; the measured process used only a few GB. A 64 GB Mac provides substantial room for this initial model and experiments, but batch size and sequence length still determine training memory. Training peak RAM, sustained concurrency, Linux/CUDA and customer hardware remain deployment measurements to perform. Mac Docker does not provide native MPS: use the native installation for Apple GPU acceleration.

## Development and deployment

```sh
.venv/bin/python scripts/check_contracts.py
.venv/bin/python -m pytest -q
.venv/bin/python -m build
```

Default tests use synthetic fixtures and download no models. Real-model checks are opt-in. [Operations](docs/operations.md) covers native macOS, Linux service and container examples, upgrades, backup and recovery. [Security](SECURITY.md) describes the trust boundary. Software is [Apache-2.0](LICENSE); separately downloaded weights and datasets retain their own terms.
