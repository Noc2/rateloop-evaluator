# Optional RateLoop connector

The evaluator works without a RateLoop account. The connector adds outbound metadata receipts, server-selected human audits and explicitly authorized human-label imports. It does not upload the evaluated text, context or evidence.

RateLoop's initial integration supports **off, shadow and paused**. Hosted receipts remain advisory and cannot reduce required human review, regardless of the evaluator's local deployment mode.

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

Import checks the export digest and current grant watermark, exact local case/input/template/model/result bindings, a conclusive human verdict, the stored pre-scoring audit, and lack of AI exposure. It retains the provenance as an authenticated overall human consensus, not an invented individual reviewer vote. Mismatches, inconclusive verdicts, exposed judgments and multi-question mappings are rejected with reasons. Repeated imports are idempotent. The subsequent snapshot builder keeps correlated cases in the same partition.

`truncated: true` means the server's bounded export did not contain every result. Do not treat it as a complete training dataset. Use narrower server export windows through a future explicitly bounded import extension rather than silently accepting missing records.

Local case deletion removes matching connector results, audit metadata and imported-label references along with retained examples. A hashed workspace/case tombstone prevents an in-flight or later request from recreating the deleted case; genuinely new work needs a new case ID. The deletion CLI also clears matching runtime cache/outbox entries.

Call `connector.close()` when the process finishes. Local backups, distributed model copies and any separately authorized human-review content have their own retention and deletion procedures; revoking a grant does not claim instant machine unlearning.
