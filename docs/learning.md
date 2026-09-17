# Private learning

Start with the exact decision your users need. Keep question text, label descriptions, language, input fields and rubric version fixed. A new question definition is a new scope. Collect human judgments before showing AI suggestions. Record the conversation or source family as `sourceGroupId`; duplicates and related cases are kept together to prevent train/test leakage.

## Feedback and consent

Use `grant --right private_training --template TEMPLATE --hours 24 --evidence 'Owner-authorized local pilot'` before collecting raw examples. This is separate from `ai_use`. A connected workspace uses short-lived grants mirrored by the connector, with exact API credential, bundle and template bindings. Do not substitute a broad manual local grant for a revoked workspace grant.

Issue a human-only credential:

```sh
rateloop-evaluator issue-reviewer-token --reviewer-id reviewer-001 --output /private/path/reviewer-001.json
```

Restart the service after changing credentials. A trusted human-review application uses that credential to POST `/v1/feedback`:

```json
{
  "workspaceId": "example-workspace",
  "evaluationId": "sha256:EXACT_INPUT_DIGEST",
  "inputCommitment": "sha256:EXACT_INPUT_DIGEST",
  "templateCommitment": "sha256:EXACT_TEMPLATE_DIGEST",
  "annotatorId": "reviewer-001",
  "labels": {"tone": "suitable"},
  "exposedToAi": false,
  "independentHuman": true
}
```

Replace the illustrative digest placeholders with the evaluated case's actual commitments. The service verifies exact correspondence and reviewer identity. The operator must authenticate the human and enforce blinding; a boolean or customer-owned signature cannot prove independence against a dishonest host. Preserve all individual labels and disagreement. Conflicting cases stay quarantined from the initial trainer rather than being converted into fabricated consensus.

## Train, calibrate and evaluate

Use at least three independent groups for a mechanical smoke test; useful quality qualification needs far more, representative of deployment. Grouped splits and immutable snapshots are enforced by the store.

```sh
rateloop-evaluator snapshot --template customer-reply-tone --version 1
rateloop-evaluator train --snapshot-id SNAPSHOT --model-dir /private/models/public-gliner --output /private/models/candidate-v1 --bundle-id customer-tone-v1 --device mps --method lora
rateloop-evaluator calibrate --snapshot-id SNAPSHOT --model-dir /private/models/candidate-v1/model --bundle-id customer-tone-v1 --device mps --output /private/calibration.json
```

Use the `modelDir` returned by training: the output directory may contain a merged model subdirectory according to the chosen method. Calibration requires that exact trained checkpoint, bundle and snapshot. Calibration records include the weight digest. Temperature fitting uses calibration groups only; final test labels never enter fitting or training.

Prepare a request file with the same immutable template and the new `modelBundleId`. Before final testing, declare the operating policy in a private JSON file, for example:

```json
{"threshold":0.95,"max_false_approval_rate":0.01,"minimum_coverage":0.3,"confidence":0.95}
```

```sh
rateloop-evaluator register --model-dir RETURNED_MODEL_DIR --request /private/candidate-request.json --snapshot-id SNAPSHOT --calibrations /private/calibration.json --selective-policy /private/policy.json --real-data
rateloop-evaluator score-test --bundle-id customer-tone-v1 --device mps --output /private/test-evidence.json
rateloop-evaluator promote --bundle-id customer-tone-v1 --template-commitment TEMPLATE_DIGEST --language en --mode selective --evidence /private/test-evidence.json
```

Use `--real-data` only for verified, consented, non-synthetic human evidence. It is a provenance declaration, not an override: promotion recomputes decisions, coverage and an exact one-sided error bound from every independent test-group representative. Zero observed mistakes in a tiny sample cannot establish a 1% error bound. Policies are fixed in signed bundles before final testing; using the same test set repeatedly for model selection still creates statistical bias, so retain fresh final validation data for each chosen release.

Local selective results are still advisory when sent to the initial RateLoop integration. Mandatory human rules and accepted human work remain in force. Test German and English separately, then relevant domains, schema changes, missing evidence, negation and adversarial cases. Large-model comparisons and customer pilot outcomes remain independent experiments, not prerequisites installed into the fast path.

## Retirement and recovery

`revoke --grant-id GRANT` invalidates dependent snapshots and retires managed derived models. `delete-case --case-id CASE` removes controlled learning records and retires affected lineage. Deployment rechecks grants and bundle validity before and after inference, including on cached requests. `rollback --template-commitment DIGEST --language en` restores only a still-authorized, unexpired prior deployment.

Stop workers before restoring an encrypted backup. Restore state, encryption and signing keys, and the exact model files together; recheck permissions and current upstream grants before serving. Never reactivate stale grants from a backup. Keep retirement and backup expiration in the customer's retention process; already copied weights require controlled deletion and replacement training.
