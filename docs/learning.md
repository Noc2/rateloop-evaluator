# Private learning

Start with the exact decision your users need. Keep question text, label descriptions, language, input fields and rubric version fixed. A new question definition is a new scope. Collect human judgments before showing AI suggestions. Record the document, conversation or other source family as `sourceGroupId`. Source groups, exact duplicates and formatting-only duplicates stay together. Semantic paraphrases are not automatically detected, so accurate source grouping remains necessary.

## Feedback and consent

Use `grant --right private_training --template TEMPLATE --hours 24 --evidence 'Owner-authorized local pilot'` before collecting raw examples. This is separate from `ai_use`. A connected workspace uses durable, versioned owner consent with exact API credential, model-bundle, template and field bindings, plus execution leases of at most 15 minutes. Renewing a lease preserves the original consent lineage. A missing lease pauses training and serving; revocation or durable consent expiry prevents reuse. Do not substitute a broad manual local grant for a revoked workspace permission.

Keep the outbound worker running during administrative training so it refreshes current permissions. Every optimizer update checks snapshot authorization; an expired or revoked lease stops further updates. Candidate registration and activation check lineage again. Changed model scope requires a new explicit owner permission; an old expired grant is never converted into indefinite consent. Website jobs retain training inputs only when permission already existed at the case's original submission time. Collect fresh cases after enabling learning.

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

## Bring your own examples

Training is optional. A compatible text model can evaluate a custom question immediately; instructions and reference
material supplied for a judgment do not update model weights. Import labeled examples only when you want a private
training or comparison dataset. Unlabeled documents belong in reference context or a labeling workflow, not fabricated
training targets.

Prepare a portable template JSON with its exact version, language, questions, label IDs/descriptions and token limit.
Upload CSV or one JSON object per line. The default columns are `case_id`, `group_id`, `text`, optional `context` and
`evidence`, and `label` for a single criterion. Several criteria use `label.QUESTION_ID` columns. Use stable case and
source-group IDs across versions. Every training row must have a valid label for each criterion. A mapping JSON can
instead name the input columns, for example:

```json
{"case_id":"record_id","group_id":"document_id","text":"summary","evidence":"source","labels":{"faithful":"expected"}}
```

Preview checks every row without retaining it or starting inference. Imports are bounded to 2 MiB and 2,000 rows;
the encrypted single-host pilot permits 50 dataset versions and 20 MiB of retained dataset examples per workspace.
Exact tokenizer limits are checked before model execution, with no silent truncation. Files, preview outputs, private
datasets and weights must stay outside Git.

```sh
rateloop-evaluator dataset-preview --file /private/summaries.csv --template-file /private/summary-template.json --format csv --mapping /private/columns.json
rateloop-evaluator grant --right private_training --template summary-faithfulness --field input.text --field input.evidence --field imported_labels --hours 24 --evidence 'Owner authorized these examples for private learning'
rateloop-evaluator dataset-import --dataset-id summary-examples --file /private/summaries.csv --template-file /private/summary-template.json --format csv --mapping /private/columns.json --provenance owner --evidence 'Owner supplied labeled examples'
rateloop-evaluator snapshot --template summary-faithfulness --version 1 --dataset-version DATASET_VERSION_ID --no-feedback
```

Private import permission is distinct from AI use and from blind human feedback permission: it covers the actual input
fields and `imported_labels`. Uploading or training does not require an unnecessary inference grant, authorize shared
contributions, or authorize public weights. Existing connected permissions remain binding; do not substitute a manual
grant to bypass a revoked or narrower workspace permission.

Imported labels retain `owner`, `ai_assisted` or `synthetic` provenance. They can train a candidate and support an
exploratory comparison against those labels. An upload cannot claim `independentHuman=true`; independent calibration
and qualification still require authenticated blind reviewer feedback. Registering imported data never creates a fake
AI receipt or human review. Conflicting labels for duplicate material stop snapshot creation until resolved.

Imports create immutable version IDs; identical retries return the same version. Select exact versions when creating
a snapshot. `--no-feedback` restricts it to those versions; without that flag, otherwise eligible authorized blind
feedback is included too. Group assignments persist across snapshots as new data arrives: held-out groups never enter
training. A later source relationship bridging frozen partitions fails closed and needs quarantine/resolution.
Revocation and deletion invalidate affected versions, snapshots and derived models.

Compare a baseline before training, then compare the candidate on the same snapshot:

```sh
rateloop-evaluator compare --snapshot-id SNAPSHOT --model baseline=/private/models/public-gliner --device mps --output /private/baseline.json
rateloop-evaluator train --snapshot-id SNAPSHOT --model-dir /private/models/public-gliner --output /private/models/summary-candidate --bundle-id summary-candidate-v1 --device mps --method lora
rateloop-evaluator compare --snapshot-id SNAPSHOT --model baseline=/private/models/public-gliner --model candidate=/private/models/summary-candidate/model --device mps --output /private/comparison.json
```

Comparisons use one representative per frozen test source group, show label provenance, per-criterion confusion counts,
false approvals/rejections, tie abstentions and Wilson intervals for agreement. Raw scores are uncalibrated; these
reports are not activation gates or probabilities of truth. Neither comparison nor candidate registration activates a
model. Repeated comparisons can leak information about the holdout into model choices; reserve fresh final evidence
before making a release-quality claim. Retraining starts from the pinned public base, not a prior private adaptation.

## Train, calibrate and evaluate

Use at least three distinct source groups for a mechanical smoke test; useful quality qualification needs far more, representative of deployment. Grouped splits and immutable snapshots are enforced by the store.

```sh
rateloop-evaluator snapshot --template summary-faithfulness --version 1
rateloop-evaluator train --snapshot-id SNAPSHOT --model-dir /private/models/public-gliner --output /private/models/candidate-v1 --bundle-id summary-candidate-v1 --device mps --method lora
rateloop-evaluator calibrate --snapshot-id SNAPSHOT --model-dir /private/models/candidate-v1/model --bundle-id summary-candidate-v1 --device mps --output /private/calibration.json
```

Use the `modelDir` returned by training: the output directory may contain a merged model subdirectory according to the chosen method. Calibration requires that exact trained checkpoint, bundle and snapshot, plus independent blind human labels. Owner-provided, AI-assisted and synthetic imports remain ineligible for that claim. Calibration records include the weight digest. Temperature fitting uses calibration groups only; final test labels never enter fitting or training.

Prepare a request file with the same immutable template and the new `modelBundleId`. Before final testing, declare the operating policy in a private JSON file, for example:

```json
{"threshold":0.95,"max_false_approval_rate":0.01,"minimum_coverage":0.3,"confidence":0.95}
```

```sh
rateloop-evaluator register --model-dir RETURNED_MODEL_DIR --request /private/candidate-request.json --snapshot-id SNAPSHOT --calibrations /private/calibration.json --selective-policy /private/policy.json --real-data
rateloop-evaluator score-test --bundle-id summary-candidate-v1 --device mps --output /private/test-evidence.json
rateloop-evaluator promote --bundle-id summary-candidate-v1 --template-commitment TEMPLATE_DIGEST --language en --mode selective --evidence /private/test-evidence.json
```

Registration creates an inactive candidate by default. Exporting its metadata does not activate it. Use explicit `promote --mode shadow` (or `register --activate` for an intentional initial activation) only after reviewing the comparison and current permissions. Use `--real-data` only for verified, consented, non-synthetic human evidence. It is a provenance declaration, not an override: promotion recomputes decisions, coverage and an exact one-sided error bound from every independent test-group representative. Zero observed mistakes in a tiny sample cannot establish a 1% error bound. Policies are fixed in signed bundles before final testing; using the same test set repeatedly for model selection still creates statistical bias, so retain fresh final validation data for each chosen release.

Local selective results are still advisory when sent to the initial RateLoop integration. Mandatory human rules and accepted human work remain in force. Test German and English separately, then relevant domains, schema changes, missing evidence, negation and adversarial cases. Large-model comparisons and customer pilot outcomes remain independent experiments, not prerequisites installed into the fast path.

## Retirement and recovery

`revoke --grant-id GRANT` invalidates dependent snapshots and retires managed derived models. `delete-case --case-id CASE` removes controlled learning records and retires affected lineage. Deployment rechecks grants and bundle validity before and after inference, including on cached requests. `rollback --template-commitment DIGEST --language en` restores only a still-authorized, unexpired prior deployment.

Stop workers before restoring an encrypted backup. Restore state, encryption and signing keys, and the exact model files together; recheck permissions and current upstream grants before serving. Never reactivate stale grants from a backup. Keep retirement and backup expiration in the customer's retention process; already copied weights require controlled deletion and replacement training.
