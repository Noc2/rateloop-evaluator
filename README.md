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
.venv/bin/rateloop-evaluator register --model-dir "$HOME/rateloop-models/gliner25" --request examples/reply-request.json
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

## Integrate RateLoop

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

The outbound `worker` command pulls authorized website cases over HTTPS, runs the pinned local model, posts fenced advisory results and synchronizes independently frozen human judgments. No public Mac port is opened. The current Alpha website sends submitted cases to its configured RateLoop-operated Mac; this is not a fully offline website. Results and provenance return as metadata; private training examples and weights stay local. [Install and run the worker](docs/operations.md#outbound-website-worker).

AI processing and private learning are separate durable owner permissions with exact credential, model and template scopes. Execution leases last at most 15 minutes and renew while connected. Enabling learning later does not retain previously queued inputs. Withdrawal retires affected models; case-erasure tombstones delete their retained examples. Legacy grants keep their original finite expiry.

The [connector guide](docs/connector.md) explains configuration, durable receipt delivery, blind audit selection and importing human outcomes. The [TypeScript client](clients/typescript/index.ts) calls your local service directly from a trusted server or agent. The [versioned interface](contracts/INTERFACE.md) and JSON schemas define cross-language commitments. The RateLoop SDK carries the same result contract.

Local AI inference does not make a cloud-connected RateLoop installation fully on-premises. Existing human-review submissions remain separately authorized and follow RateLoop's content policy. A complete customer-hosted human-review platform is outside this evaluator repository.

## Learn from opted-in human ratings

Grant `private_training` separately before retaining any examples. Issue separate, identity-bound reviewer credentials with `issue-reviewer-token`; inference agents cannot submit human feedback. Feedback must bind the exact input and template commitments. AI-exposed, duplicated, conflicting or non-independent labels are quarantined. An overall human verdict cannot silently become several criterion labels.

The [learning guide](docs/learning.md) covers the sequence: collect authorized independent labels → create a grouped snapshot → train from the public base → calibrate on independent groups → register an immutable candidate and operating policy → score the final test groups → promote only if its measured gate passes. Retrain from the public base using currently authorized data for each release; training on an already adapted private checkpoint is deliberately rejected until recursive parent lineage is implemented.

`ai_use`, `private_training`, `shared_contribution` and `public_weight_distribution` are independent rights. Private training does not authorize shared training or public weights. Public weights do not require publishing raw examples. Revocation retires affected managed datasets and models; it does not promise instant unlearning from copied weights.

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
