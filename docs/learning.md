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

Guiding examples are disclosed model inputs, even when a blind human later rates the same text. A training, calibration or test row that repeats an example after whitespace normalization blocks snapshot creation and loading, calibration, scoring and qualification. Existing stored snapshots are checked too; rows and human labels are never silently removed or relabeled. Keep guiding examples separate from evaluation evidence. This exact-text check does not detect semantic paraphrases.

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

Training preserves exactly the inference question order, label IDs, descriptions and guiding examples. The pinned upstream trainer's synthetic-label augmentation is disabled: it could reinsert an original correct label as a negative option after aliasing it. Classification targets also bind to the complete schema, avoiding an upstream prefix-ID match that could attach another question's labels. This changes preprocessing only; the optimizer loss, authorization checks and validation gate remain unchanged. Manifests identify this policy as `immutable-inference-schema-v1`.

Use at least three distinct source groups for a mechanical smoke test; useful quality qualification needs far more, representative of deployment. Grouped splits and immutable snapshots are enforced by the store.

The website's version-2 training recipe uses a bounded validation run. It reserves a fixed 20% of the existing training source groups for checkpoint selection, keeping calibration and final-test partitions separate. The encrypted store freezes these validation assignments across dataset versions and worker restarts; a later source relationship joining optimizer and validation groups fails closed instead of changing either role. Each optimizer and validation partition needs at least five source groups for every declared label; sparse data fails with a request for more examples. This is a minimum for running the procedure, not evidence that a dataset is representative. Plan several hundred carefully labeled examples for an initial task pilot, then inspect class and source coverage.

The optimizer balances source-group label combinations, uses a fixed learning rate policy, evaluates every 25 steps, stops after three checks without improved balanced agreement, and never exceeds the advertised step cap. Checkpoint selection requires higher balanced agreement than the untrained base with no observed per-class recall regression. If none qualifies, training returns `validation_not_improved`, preserves the original evaluator and publishes no candidate. The private selection report records the baseline, all checkpoint metrics, and rejection reason. An accepted candidate is still experimental: separate final-test comparison and explicit owner activation remain necessary.

For local CLI training, add `--max-steps 200 --learning-rate .0001 --validation-fraction .2`. The Python primitive is `TrainOptions(max_steps=200, learning_rate=1e-4, validation_fraction=.2, validation_interval=25, early_stopping_patience=3, min_validation_per_label=5)`. Its selected step, optimizer sampling policy, actual training IDs and validation-only IDs are committed in the model's training manifest. A zero validation fraction retains the older fixed-budget mechanical path and is labeled `fixed_budget_smoke`; it cannot establish an improvement. No optimizer step or validation prediction bypasses current snapshot authorization.

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

## Reproducible public quality experiments

`scripts/run_public_quality.py` prepares and runs a bounded, offline HelpSteer2 experiment using a previously provisioned model. Obtain the public NVIDIA `train.jsonl.gz` and `validation.jsonl.gz` files explicitly from revision `990b2711a36180dd19d9c94b8627844866f8982a`; the tool refuses bytes that differ from the checked-in hashes. NVIDIA publishes these human reference labels under CC-BY-4.0: retain [the source attribution](https://huggingface.co/datasets/nvidia/HelpSteer2) and identify the filtering/binary conversion when sharing results. Raw source text, private stores, keys and checkpoint files stay outside Git.

```sh
python scripts/run_public_quality.py --train /private/HelpSteer2/train.jsonl.gz --test /private/HelpSteer2/validation.jsonl.gz --model-dir /private/models/public-gliner --criterion helpfulness --device mps --output /private/quality-helpfulness
```

Run `--criterion correctness` separately. Defaults freeze 400 development prompt groups and up to 200 final groups before prediction. Scores 0–1 and 3–4 become the two labels; ambiguous score 2, duplicate prompts, repeated normalized answers and inputs beyond the exact token budget are excluded and counted. Development snapshots reserve separate optimizer, validation and calibration groups. The official validation split is used only for final testing. A temperature fit uses the calibration partition only; final Brier score and calibration error remain descriptive external-label measurements, not calibrated production-confidence claims.

If an earlier experiment already exposed that final set, `--previous-experiment /private/quality-helpfulness` reserves a new high-hash tail of upstream training prompts, excludes it from development, and requires the original development commitment to remain identical. Its report explicitly identifies that alternative final source. Do not repeatedly tune against either final set. A rejected validation run exports no candidate and skips candidate final-test scoring. Public-data experiments never change website activation or qualify a workspace for autonomous decisions. See the [original measurements and their preprocessing limitation](verification.md#public-quality-and-validation-gates--28-september-2026) and the [fixed-code attempts](verification.md#post-correction-public-regression-attempts).

## General-request benchmark and qualification diagnostics

`scripts/run_general_benchmark.py` runs the broader diagnostic path without modifying a worker, model registration or
production default. Explicitly download the pinned HelpSteer2 training file described above, then freeze one response
per normalized source prompt before any model predictions:

```sh
python scripts/run_general_benchmark.py prepare-public --source-gzip /private/HelpSteer2/train.jsonl.gz --criterion helpfulness --maximum-groups 600 --output /private/general-helpfulness
python scripts/run_general_benchmark.py run --manifest /private/general-helpfulness/manifest.json --rows /private/general-helpfulness/rows.json --model-dir /private/models/public-gliner --device mps --output /private/general-helpfulness-base
```

The default threshold is fixed at 0.9 before final predictions. `run` writes both operating-point commitments and any
temperature fits before observing final scores. It uses calibration groups only for fitting; test groups are never used
for fitting or threshold selection. Calibrators with fewer than 20 completed groups or only one reference class are
omitted. This is a diagnostic minimum, not enough evidence to qualify confidence. Files are exclusive, owner-only writes;
use a new output directory for another predeclared candidate. Repeated inspection or selection on the same holdout needs
fresh evidence before a release-quality claim. `--backend gliclass` uses the separately provisioned GLiClass runtime and
weights. No command downloads models or creates a hosted fallback.

Unlike the earlier short-input benchmark, the general experiment retains long selected requests. Overflow, unsupported,
failed, partial, unperformed and withheld checks remain in coverage denominators. Proper scores explicitly report their
completed-case denominator. Reports include class recall/precision, false rejections, accepted-case false acceptance,
exact one-sided bounds, useful coverage, NLL/Brier/ECE, reliability-bin intervals, latency and full logical-evaluation
API cost. The local runner reports warm local processing latency and model-load time separately; it does not measure
website/network/hosted queue latency, concurrent capacity, hardware or electricity cost. The zero API cost is specific
to this local run. Slice-wide claims use a Bonferroni-adjusted bound alongside each ordinary 95% bound. Public diagnostics
always report `qualified: false` and `quality_gate: false`, even if statistical targets are met.

For additional licensed or authorized EN/DE data, `freeze --rows ROWS.json --sources SOURCES.json --output MANIFEST.json`
accepts the same row format emitted by `prepare-public`: stable evaluation/group IDs, input, exact template, labels,
source ID, family and language. Families are `request_following`, `source_faithfulness`, `knowledge`, `writing`,
`code_math` and `unsupported`. A source declares its revision, content SHA-256, license, attribution, URL and label
provenance. Public human labels retain `external_human`; owner, AI-assisted and synthetic sources retain those origins.
The public adapter's family is the declared rubric scope, not a claim that HelpSteer2 supplies an independent six-family
task classification. Its absent German and source-faithfulness slices remain visibly empty.

Related source IDs must link paraphrases, translations, conversations and document families. Normalized prompt/document
and exact/formatting-equivalent input aliases also group automatically. The frozen manifest is content-free and records
50% development, 25% calibration and 25% test hash assignments; actual small-slice counts vary. Reserve the existing fixed
20% validation fraction within development for candidate selection. Pass `--previous MANIFEST.json` when extending a
cohort to preserve assignments; a newly discovered relationship crossing frozen roles fails closed. Do not force exact
counts by moving held-out material into development. Semantic duplicates still require explicit source links.

`packet --manifest MANIFEST.json --rows ROWS.json --output BLIND.json` exports instructions, inputs and anchored rubrics
without imported labels or AI predictions. Two eligible reviewers should independently annotate, preserving disagreement
for blinded adjudication. The packet neither authenticates reviewers nor upgrades imports. Actual independent feedback
must enter through the authenticated review path described earlier; a file's provenance declaration is not proof.

`point --manifest MANIFEST.json --configuration CONFIG.json --output POINT.json` freezes an external route's model,
weights/tokenizer, runtime, quantization, evidence policy, optional calibrator and operating threshold. `report --manifest
MANIFEST.json --rows ROWS.json --point POINT.json --observations OBSERVATIONS.json --output REPORT.json` accepts one bound
observation for every test representative, including operational skips. This supports comparing an explicitly provisioned
local judge or separately authorized external experiment without manufacturing probabilities from generated confidence
text. Only normalized complete distributions may supply probabilistic diagnostics; incomplete checks cannot masquerade
as completed scores. Whole-route API costs include retries and child checks. No report activates or qualifies anything.

The production confidence path remains the signed `BundleRegistry` promotion gate over authenticated, blinded reference
snapshots, exact weight-bound calibration, immutable operating policy and unexpired evidence. General diagnostics add
minimum-accepted-group, per-family/language coverage and simultaneous-bound checks; they do not weaken existing gates.
External public data first, plus missing target-population evidence, means native confidence remains unvalidated.

## Retirement and recovery

`revoke --grant-id GRANT` invalidates dependent snapshots and retires managed derived models. `delete-case --case-id CASE` removes controlled learning records and retires affected lineage. Deployment rechecks grants and bundle validity before and after inference, including on cached requests. `rollback --template-commitment DIGEST --language en` restores only a still-authorized, unexpired prior deployment.

Stop workers before restoring an encrypted backup. Restore state, encryption and signing keys, and the exact model files together; recheck permissions and current upstream grants before serving. Never reactivate stale grants from a backup. Keep retirement and backup expiration in the customer's retention process; already copied weights require controlled deletion and replacement training.
