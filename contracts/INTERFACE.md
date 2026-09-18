# Implementation contract

Portable request/result models live in `rateloop_evaluator.protocol`; JSON Schemas and TypeScript validators are derived
from their published schema. Wire fields use camelCase. No private product services are copied into this repository.

`EvaluationRequest`: schemaVersion `rateloop.evaluator.request.v1`, workspaceId, caseId, sourceGroupId (nullable, conversation/document identity), idempotencyKey,
template (id, version, language, questions[{id,text,labels[{id,description}],passLabels}], maxTokens),
input {text,context,evidence}, modelBundleId, deadlineMs.

`EvaluationResult`: schemaVersion `rateloop.evaluator.result.v1`, workspaceId, caseId, modelBundleId,
inputCommitment, templateCommitment, outcome pass/fail/uncertain, abstainReason, criteria
[{questionId,label,rawScores,probabilities,calibrationId}], durationMs, observedAt, resultCommitment.

Scores are advisory without valid bundle/template/language calibration. Policy controls human review separately.
Input commitment binds workspace, case, source group, template, input and model bundle, excluding retry key and deadline.
Commitments use SHA-256 of UTF-8 `domain + "\n"` followed by RFC8785 canonical JSON bytes. Domains are
`rateloop.evaluator.input.v1`, `rateloop.evaluator.template.v1` and `rateloop.evaluator.result.v1` respectively.
Result commitment covers every result field except resultCommitment. Non-finite and unsafe integers are rejected.

Backend API: `predict(text: str, questions: list[dict]) -> dict[str, dict[str, float]]`; each question has id, text,
labels [{id,description}], passLabels. Return per-question scores keyed by stable label ID; never silently drop labels.
`count_tokens(text: str, questions: list[dict]) -> int` includes schema tokens. Constructor accepts an explicitly local
model directory and device, loads lazily, no downloads. Model provisioning is a separate explicit command.

Service responsibilities: strict schema and token limits, tenant tokens, deadline/idempotency, calibration, result storage,
feedback and optional approved-metadata outbox. Learning/registry modules expose their own small documented functions;
the service uses those functions rather than duplicating grant checks. Model/backend/training checks require real
local hardware before claiming support. Synthetic predictors are tests only and never selectable by a runtime flag.

The outbound website protocol uses `/api/assurance/v2/evaluations/jobs/claim`, followed by lease-scoped
`/jobs/{jobId}/content`, `/heartbeat`, `/complete` and `/fail`. Claims contain metadata only, including frozen
`reviewMode: "ai" | "ai_and_human"`. Content repeats that exact mode and returns the portable request, original
`createdAt`, `retainForTraining`, audit and agent identity. An omitted mode in either response means `ai_and_human`;
explicit null, unknown values, or a claim/content mismatch are rejected. Human-only cases create no evaluator job.
Job leases last 120 seconds and fencing headers bind receipts to the current worker: `X-Evaluator-Job`,
`X-Evaluator-Worker`, `X-Evaluator-Lease`.

For `ai_and_human`, the audit remains mandatory, selected with probability 10000 basis points, unexposed and
`server_enforced`; results stay hidden until the independent answer freezes. For `ai`, content requires explicit
`audit: null` and `retainForTraining: false`. Its result can be shown immediately, remains advisory, and never
creates an independent human training reference. The worker's completed result has `humanReviewRequired: false`
only for an explicit AI-only job. Review selection does not change the portable result schema, calibration,
uncertainty, consent, model qualification or immutable input commitment.

Grant synchronization can include immutable durable `consents` and a distinct recipient-bound `authorizationLease`
of at most 900 seconds. Stable consent revisions, explicit fields, template commitments and model bundle IDs determine
training lineage. Lease renewal cannot widen those scopes or revive revoked consent. Explicit `deletedCases` tombstones
identify the case records to erase; credential rejection alone is not a deletion instruction. See the connector guide
for legacy grant compatibility and the operations guide for workspace-erasure boundaries.

Portable connector audit calls require the existing `frozen_question_hash` from the human-review schema, not just the
question prompt. Source and suggestion hashes separately bind their exact UTF-8 content. Website workers also enforce
the frozen `customer-reply-approval` EN/DE templates; template translations remain separate snapshot scopes.
