# Implementation contract

Portable request/result models live in `rateloop_evaluator.protocol`; JSON Schemas and TypeScript validators are derived
from their published schema. Wire fields use camelCase. No private product services are copied into this repository.

`EvaluationRequest`: schemaVersion `rateloop.evaluator.request.v1`, workspaceId, caseId, idempotencyKey,
template (id, version, language, questions[{id,text,labels[{id,description}],passLabels}], maxTokens),
input {text,context,evidence}, modelBundleId, deadlineMs.

`EvaluationResult`: schemaVersion `rateloop.evaluator.result.v1`, workspaceId, caseId, modelBundleId,
inputCommitment, templateCommitment, outcome pass/fail/uncertain, abstainReason, criteria
[{questionId,label,rawScores,probabilities,calibrationId}], durationMs, observedAt, resultCommitment.

Scores are advisory without valid bundle/template/language calibration. Policy controls human review separately.
Input commitment binds workspace, case, template, input and model bundle, excluding retry key and deadline.
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
