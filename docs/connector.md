# Optional RateLoop connector

The evaluator works without a RateLoop account. The rating connector adds outbound metadata receipts, server-selected human audits and explicitly authorized human-label imports. Rating receipts do not upload the evaluated text, context or evidence. The separately enabled generation worker receives Chat inputs and sends generated text back to the connected workspace. Local inference does not remove the workspace’s existing handling of those inputs and outputs.

RateLoop's evaluator connection supports **off, shadow and paused**. Separately, each website case selects **Human**, **AI** or **AI + human**. AI-only returns an advisory result without creating human review; AI + human withholds that result until the independent human answer freezes. Hosted receipts cannot automatically satisfy required human review, regardless of the evaluator's local deployment mode.

## Register a model with RateLoop

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

The outbound `worker` command pulls authorized website cases over HTTPS, runs the pinned local model, posts fenced advisory results and synchronizes independently frozen human judgments. No public Mac port is opened. The website can route permitted cases to its configured RateLoop-operated worker or the explicitly selected customer device; this is not a fully offline website. An always-on Linux CPU worker uses the same outbound protocol and shared English/German weights. [Hosted setup and readiness checks](operations.md#always-on-hosted-cpu-worker). Results and provenance return as metadata; private training examples and weights stay local. [Install and run the worker](operations.md#outbound-website-worker).

AI processing and private learning are separate durable owner permissions with exact credential, model and template scopes. Execution leases last at most 15 minutes and renew while connected. Enabling learning later does not retain previously queued inputs. Withdrawal retires affected models; case-erasure tombstones delete their retained examples. Legacy grants keep their original finite expiry.

The sections below explain configuration, durable receipt delivery, blind audit selection and importing human outcomes. The [TypeScript client](../clients/typescript/index.ts) calls your local service directly from a trusted server or agent. The [versioned interface](../contracts/INTERFACE.md) and JSON schemas define cross-language commitments. The RateLoop SDK carries the same result contract.

Local AI inference does not make a cloud-connected RateLoop installation fully on-premises. Existing human-review submissions remain separately authorized and follow RateLoop's content policy. A complete customer-hosted human-review platform is outside this evaluator repository.

## Configuration

Create `RateLoopConnector` with:

| Argument | Meaning |
| --- | --- |
| `base_url` | Explicit HTTPS origin, without a path, query, credentials or fragment. |
| `api_key` | Workspace API credential, read from a protected secret file. |
| `api_key_id` | The exact credential recipient ID shown by RateLoop. |
| `workspace_id` | The workspace that owns both the local cases and server credential. |
| `agent_id`, `agent_version_id` | The registered agent identity for these receipts. |
| `learning`, `runtime` | Initialized encrypted `LearningStore` and `RuntimeStore` instances. |
| `metadata_upload_enabled` | Defaults to `False`. Set `True` only after approving the metadata destination and purpose. |
| `allow_insecure_loopback` | Defaults to `False`. Development-only permission for an HTTP loopback origin. |

Use the workspace owner's RateLoop evaluator settings to register the exact bundle/template, enable shadow mode and issue learning grants to this API-key ID. The API key needs `evaluation:read` to synchronize permissions and import labels, `telemetry:write` to upload receipts, and `review:decide` for audits.

Keep credentials outside source control. The connector disables proxy/environment configuration and redirect following. It does not accept a server-supplied destination. HTTP response bodies are limited to 10 MB of decoded data while streaming.

```python
from rateloop_evaluator.connector import RateLoopConnector
from rateloop_evaluator.learning import read_secret

connector = RateLoopConnector(
    base_url="https://www.rateloop.ai",
    api_key=read_secret("/private/operator/rateloop-api-key").decode(),
    api_key_id="workspace-credential-id",
    workspace_id="workspace-id",
    agent_id="registered-agent-id",
    agent_version_id="registered-agent-version-id",
    learning=learning_store,
    runtime=runtime_store,
    metadata_upload_enabled=True,
)
```

The example origin is RateLoop's branded Alpha. `learning_store` and `runtime_store` are existing customer-owned local stores; the connector never creates a cloud copy of their contents.

Website jobs carry their frozen `reviewMode` in both the authenticated claim and content. Only `ai` explicitly omits human review; `ai_and_human` and legacy omissions require a selected, server-blinded audit. This includes sampled comparisons in a saved AI/human balance. The worker preserves the server's `random` or `mandatory` selection kind and exact integer probability (1–10,000 basis points) through retries and human-label import. Human-only does not create an evaluator job. AI-only content must contain `audit: null` and `retainForTraining: false`. The worker checks the same committed request, consent, model, agent identity and execution lease for either AI choice, and refuses a mode change on retry. No audit or human training reference is manufactured for AI-only cases; a sampled comparison still needs separate learning permission to retain training inputs.

## Synchronize learning permissions

### Command-line configuration

The CLI reads a separate mode-0600 JSON configuration file. Use actual IDs from the workspace owner interface and an existing registered agent. This example uses placeholders, not working credentials:

```json
{
  "baseUrl": "https://www.rateloop.ai",
  "apiKey": "REPLACE_WITH_WORKSPACE_API_KEY",
  "apiKeyId": "workspace-credential-id",
  "agentId": "registered-agent-id",
  "agentVersionId": "registered-agent-version-id",
  "metadataUploadEnabled": true
}
```

```sh
chmod 600 /private/connector.json
rateloop-evaluator connect-sync --config /private/connector.json
rateloop-evaluator connect-evaluate --config /private/connector.json --request /private/request.json --review-context /private/review-context.json --frozen-question-hash SHA256_QUESTION_COMMITMENT
rateloop-evaluator connect-flush --config /private/connector.json
rateloop-evaluator connect-import-labels --config /private/connector.json --grant-id GRANT --question-id reply_ready --template-commitment DIGEST --positive-label ready --negative-label needs_revision
rateloop-evaluator connect-release --config /private/connector.json --input-commitment DIGEST
```

`connect-evaluate` checks the current workspace mode and grants before scoring, using the local credential from the selected state directory. `--client-credentials` can select a different protected local credential; `--endpoint` can address a customer TLS evaluator. It does not contact an arbitrary cloud inference provider. The review-context JSON follows the metadata shape below. Selected AI results remain withheld until explicit release; complete the independently authorized human-review workflow and import first. `connect-flush` sends the queued receipts; running these commands does not enable a background task automatically.

```python
status = connector.sync_grants()
```

The connector checks the authenticated response's workspace and API-key recipient, then maps the server's distinct permissions:

| RateLoop permission | Local right |
| --- | --- |
| `ai_use` durable consent | `ai_use` |
| `private_learning` | `private_training` |
| `shared_contribution` | `shared_contribution` |
| `public_weights` consent or legacy `publicWeightsAllowed` flag | `public_weight_distribution` |

Only explicit `ai_use` permission enables inference; learning and sharing permissions do not imply it. Shared contribution does not imply private training. Field permissions preserve the distinction between input, context, evidence and human labels. Retaining input does not itself authorize storing or training on labels.

Legacy grants remain bound to their original scope and at-most-24-hour expiration. The website worker uses consents[] with immutable consentId/revision, explicit modelBundleIds, templateCommitments, fields, purpose, processingLocation, modelFamilyId and recipient. Separate authorizationLease values last at most 15 minutes. Renewals update execution authorization without changing consent lineage or widening scope; changed scope requires an explicit new revision. Revocation watermarks cannot move backwards. Revocation invalidates managed descendants; lease expiry pauses their use until a valid renewal. Consent and legacy grant namespaces remain separate.

An offline machine cannot know about remote revocation immediately. Worker authorization stops after at most 15 minutes; legacy grants retain their original deadline. Reconnecting checks the current watermark and explicit case-erasure tombstones. Pausing ratings does not itself revoke an independently issued learning grant. Explicit local owner grants remain separate from mirrored server grants.

## Select an audit before scoring

Call `run_with_audit` before calling the local evaluator for that case:

```python
outcome = connector.run_with_audit(
    request,                  # Validated EvaluationRequest containing local text.
    evaluate=local_evaluate,   # Callback: EvaluationRequest -> EvaluationResult.
    review_context=metadata,  # Existing RateLoop policy/execution metadata only.
    frozen_question_hash=existing_human_question_commitment,
    allow_offline=False,
)
```

Supply the existing frozen human-review question commitment, covering its prompt, labels and review semantics. Do not substitute a hash of the prompt alone. Source and suggestion hashes are computed from the exact request context/text bytes and checked by the human handoff.

`review_context` accepts only the existing policy identifiers, policy version, workflow/risk identifiers, audience commitment and bounded execution metadata. It rejects raw summaries, rubrics, free-form context and arbitrary nested fields. The audit API can open an existing human-review opportunity; this connector does not upload the human-review task's content. Any separate content handoff uses the existing authorized RateLoop review workflow.

A minimal metadata shape is shown below. Replace the identifiers with the existing agent policy and workflow values; placeholders do not create a policy or agent registration. The provider/model fields describe the agent whose output is being reviewed.

```python
metadata = {
    "policyId": "existing-policy-id",
    "policyVersion": 1,
    "workflowKey": "customer-reply",
    "riskTier": "low",
    "audiencePolicyHash": "sha256:" + "a" * 64,  # Existing policy commitment.
    "metadataComplete": True,
    "execution": {
        "externalExecutionId": "execution-001",
        "status": "completed",
        "primarySpanId": "generation-001",
        "generationSpans": [{
            "spanId": "generation-001",
            "role": "primary",
            "provider": "local",
            "requestedModel": "registered-agent-model",
        }],
    },
}
```

The order is:

1. Request a random audit selection from RateLoop using the committed case identity.
2. Persist the server's selection before scoring.
3. Evaluate locally and verify the returned case, model, input and template commitments.
4. Queue a metadata receipt using a stable idempotency key.
5. Return the AI result only when the case was not selected for an independent audit.

For selected cases, `outcome["result"]` is `None`, `awaitingIndependentHuman` is `True`, and `audit` contains the existing human-review opportunity. Keep the AI answer away from reviewers. Import their completed independent judgment before explicitly calling:

```python
result = connector.release_result(request.input_commitment())
```

Release records AI exposure. Legacy connector-attested labels must be imported before release. Website jobs instead use server-enforced causal records: a human answer frozen before release remains independent when imported afterward. The importer verifies the frozen question, source and suggestion hashes, the independent audit and the order of reviewFrozenAt/resultsReleasedAt. Answers created after exposure are rejected.

Blinding is **connector-attested**, not a guarantee about what a reviewer viewed elsewhere. Do not show a reviewer the receipt dashboard, AI criteria or another copy of the result before collecting their judgment. A caller bypassing this orchestration cannot claim its audits were independently blind.

With `allow_offline=True`, a temporary server outage allows local evaluation under still-valid local permissions and an enabled workspace state synchronized within the previous 24 hours. A known pause or missing state cannot be bypassed. The result is explicitly marked with `blindingAssurance: "none"`; it cannot later be relabeled as a pre-scoring blind audit. Authentication, invalid scope and malformed server responses fail closed rather than being treated as an offline outage.

## Send receipts

`run_with_audit` already queues its receipt. For a separately evaluated case, call `queue_result(result)` explicitly. Queueing a result does not establish an independent audit.

```python
receipt_key = connector.queue_result(result)
report = connector.flush(limit=20)
```

Only the validated v2 receipt is uploaded: agent identity plus the evaluator's structured result and commitments. Text, context, evidence and label descriptions are excluded. Upload remains disabled unless explicitly opted in.

The encrypted outbox survives process restarts. Temporary failures retry with bounded exponential backoff and the same idempotency key. Credentials are rechecked by the server. The connector verifies acknowledgment identity, receipt commitment and advisory policy before removing an accepted item. Receipts outside the server's 24-hour ingest window or permanently invalid payloads become local dead-letter metadata rather than being retried indefinitely. Another connector's namespaced queue entries are not sent or removed.

## Import independent human labels

The current RateLoop export contains **overall human verdicts**, not labels for every criterion. Automatic import therefore supports exactly one explicitly mapped question:

```python
report = connector.fetch_and_import_labels(
    "server-learning-grant-id",
    question_id="reply_ready",
    template_commitment=request.template_commitment(),
    outcome_labels={"positive": "ready", "negative": "needs_revision"},
)
```

Choose this mapping only when the overall human review actually answers that question. A narrow tone criterion is not interchangeable with an overall approval unless the review itself was defined that way. Multi-question evaluations require separately authenticated criterion-level adjudication; this connector does not invent those labels.

Import checks the export digest and current grant watermark, exact local case/input/template/model/result bindings, a conclusive human verdict, the stored pre-scoring audit, and lack of AI exposure. It retains the provenance as an authenticated overall human consensus, not an invented individual reviewer vote. AI-only cases cannot contribute human labels, including through the local feedback API. Mismatches, inconclusive verdicts, exposed judgments and multi-question mappings are rejected with reasons. Repeated imports are idempotent. The subsequent snapshot builder keeps correlated cases in the same partition.

`truncated: true` means the server's bounded export did not contain every result. Do not treat it as a complete training dataset. Use narrower server export windows through a future explicitly bounded import extension rather than silently accepting missing records.

Local case deletion removes matching connector results, audit metadata and imported-label references along with retained examples. A hashed workspace/case tombstone prevents an in-flight or later request from recreating the deleted case; genuinely new work needs a new case ID. The deletion CLI also clears matching runtime cache/outbox entries.

Call `connector.close()` when the process finishes. Local backups, distributed model copies and any separately authorized human-review content have their own retention and deletion procedures; revoking a grant does not claim instant machine unlearning.

## Local generation identity

The paired connector stores an optional generation runtime separately from rating bundles. Its `rateloop.generation-model.v1` identity is registered through `/api/assurance/v2/generation/models`. Jobs carry the same immutable identity and commitment; only the operator-selected loopback origin in private local configuration is callable.

`weightDigest` is the first local `FROM` blob SHA-256 reported by Ollama. `templateDigest` uses the existing RFC8785 commitment function under `rateloop.ollama.template.v1` with exactly these keys: `manifestDigest` (normalized `sha256:` prefix), `template`, `system`, `parameters`, `messages`. Missing string fields default to `""`; missing messages default to `[]`. The manifest commitment also binds auxiliary model blobs, renderer and parser metadata. This identifies installed artifacts; it is not a quality qualification.

The completion outbox is a separate encrypted database from rating receipts. It saves final text and the lease only for exact retry after a lost acknowledgment; no prompt or partial stream is saved. A revoked/expired lease cannot authorize new inference or completion. See the README for the single `connect` / `start` / `install` workflow.


### Experimental judge enrollment

Use `connect --judge-model <already-installed-model>` to register the fixed EN/DE
label-only approval judge for a device that permits AI ratings. It can be combined with
`--generation-model`, independently authorized by the pairing code. GLiNER `--model-dir`
is optional. The worker uses exactly the returned device worker ID and immutable bundle
allowlist. Label-only judging cannot register custom-question or native-qualified GLiNER
capabilities, emit synthetic numeric confidence, or retain training inputs.
See `contracts/INTERFACE.md` for result v2 semantics. Model-specific licenses remain
operator responsibilities; pairing does not grant distribution rights.
