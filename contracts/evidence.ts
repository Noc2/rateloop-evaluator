// SPDX-License-Identifier: Apache-2.0
// Additive evidence v2. A v1 result and its commitment remain unchanged.
import { commitment, type EvaluationRequest, type EvaluationResult } from "./evaluator.ts";
import { isSuppliedMaterialTemplate } from "./source-evidence.ts";
export type RuntimeIdentity = {
  modelId: string | null; modelRevision: string | null; weightsCommitment: string | null;
  tokenizerCommitment: string | null; runtime: string; precision: string | null; scoreAdapter: string;
};
export type EvidenceCheck = {
  id: string; questionId: string | null; kind: "criterion" | "supplied_material" | "factual_claims" | "code_execution";
  state: "completed" | "not_checked" | "not_applicable" | "pending" | "withheld" | "failed";
  judgment: "meets" | "does_not_meet" | "insufficient_evidence" | null; reasonCode: string | null;
  coverage: {status: "complete" | "partial" | "none" | "unknown"; checkedUnits: number; totalUnits: number | null};
  sourceRefs: {sourceId: string; sourceCommitment: string; passageCommitment: string; startByte: number; endByte: number}[];
  calibration: {status: "not_validated" | "mapping_registered"; calibrationId: string | null};
};
export type EvaluationEvidence = {
  schemaVersion: "rateloop.evaluator.evidence.v2"; workspaceId: string; caseId: string; modelBundleId: string;
  inputCommitment: string; templateCommitment: string; resultCommitment: string; answerCommitment: string;
  language: "en" | "de"; observedAt: string; identity: RuntimeIdentity; checks: EvidenceCheck[]; evidenceCommitment: string;
};
const domain = "rateloop.evaluator.evidence.v2";
function invalid(): never { throw new Error("Invalid evaluator evidence"); }
const isId = (v: unknown): v is string => typeof v === "string" && /^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$/u.test(v);
const isDigest = (v: unknown): v is string => typeof v === "string" && /^sha256:[a-f0-9]{64}$/u.test(v);
function object(v: unknown, keys: string[]): Record<string, unknown> {
  if (!v || typeof v !== "object" || Array.isArray(v) || ![Object.prototype, null].includes(Object.getPrototypeOf(v))) invalid();
  const record = v as Record<string, unknown>;
  if (Object.keys(record).length !== keys.length || keys.some(key => !Object.hasOwn(record, key))) invalid();
  return record;
}
const integer = (v: unknown, max: number) => Number.isSafeInteger(v) && Number(v) >= 0 && Number(v) <= max;
const text = (v: unknown, max: number) => typeof v === "string" && v.length > 0 && Array.from(v).length <= max;
export function parseEvaluationEvidence(input: unknown): EvaluationEvidence {
  const value = object(input, ["schemaVersion", "workspaceId", "caseId", "modelBundleId", "inputCommitment", "templateCommitment", "resultCommitment", "answerCommitment", "language", "observedAt", "identity", "checks", "evidenceCommitment"]);
  if (value.schemaVersion !== domain || !["en", "de"].includes(String(value.language))) invalid();
  for (const key of ["workspaceId", "caseId", "modelBundleId"]) if (!isId(value[key])) invalid();
  for (const key of ["inputCommitment", "templateCommitment", "resultCommitment", "answerCommitment", "evidenceCommitment"]) if (!isDigest(value[key])) invalid();
  if (typeof value.observedAt !== "string" || !/^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z$/u.test(value.observedAt) || !Number.isFinite(Date.parse(value.observedAt)) || new Date(value.observedAt).toISOString() !== value.observedAt) invalid();
  const identity = object(value.identity, ["modelId", "modelRevision", "weightsCommitment", "tokenizerCommitment", "runtime", "precision", "scoreAdapter"]);
  for (const key of ["weightsCommitment", "tokenizerCommitment"]) if (identity[key] !== null && !isDigest(identity[key])) invalid();
  if (identity.modelId !== null && !text(identity.modelId, 200) || identity.precision !== null && !text(identity.precision, 40) || identity.modelRevision !== null && (typeof identity.modelRevision !== "string" || !/^[a-f0-9]{40,64}$/u.test(identity.modelRevision))) invalid();
  if (!text(identity.runtime, 160) || !text(identity.scoreAdapter, 160)) invalid();
  if (!Array.isArray(value.checks) || !value.checks.length || value.checks.length > 24) invalid();
  const ids = new Set<string>(); const questionIds = new Set<string>();
  for (const entry of value.checks) {
    const check = object(entry, ["id", "questionId", "kind", "state", "judgment", "reasonCode", "coverage", "sourceRefs", "calibration"]);
    if (!isId(check.id) || ids.has(check.id)) invalid(); ids.add(check.id);
    if (check.questionId !== null && !isId(check.questionId) || check.reasonCode !== null && !isId(check.reasonCode)) invalid();
    if (check.questionId !== null) { if (questionIds.has(String(check.questionId))) invalid(); questionIds.add(String(check.questionId)); }
    if (!["criterion", "supplied_material", "factual_claims", "code_execution"].includes(String(check.kind)) || !["completed", "not_checked", "not_applicable", "pending", "withheld", "failed"].includes(String(check.state))) invalid();
    if (check.judgment !== null && !["meets", "does_not_meet", "insufficient_evidence"].includes(String(check.judgment))) invalid();
    if ((check.state === "completed") !== (check.judgment !== null) || check.kind === "criterion" && check.questionId === null) invalid();
    const coverage = object(check.coverage, ["status", "checkedUnits", "totalUnits"]);
    if (!["complete", "partial", "none", "unknown"].includes(String(coverage.status)) || !integer(coverage.checkedUnits, 100_000) || coverage.totalUnits !== null && !integer(coverage.totalUnits, 100_000)) invalid();
    if (coverage.totalUnits !== null && Number(coverage.checkedUnits) > Number(coverage.totalUnits)) invalid();
    if (coverage.status === "complete" && (!coverage.checkedUnits || coverage.totalUnits !== coverage.checkedUnits) || coverage.status === "none" && coverage.checkedUnits !== 0 || coverage.status === "partial" && (!coverage.checkedUnits || coverage.totalUnits !== null && Number(coverage.checkedUnits) >= Number(coverage.totalUnits))) invalid();
    const calibration = object(check.calibration, ["status", "calibrationId"]);
    if (!["not_validated", "mapping_registered"].includes(String(calibration.status)) || calibration.calibrationId !== null && !isId(calibration.calibrationId) || (calibration.status === "mapping_registered") !== (calibration.calibrationId !== null)) invalid();
    if (!Array.isArray(check.sourceRefs) || check.sourceRefs.length > 20) invalid();
    for (const item of check.sourceRefs) {
      const ref = object(item, ["sourceId", "sourceCommitment", "passageCommitment", "startByte", "endByte"]);
      if (!isId(ref.sourceId) || !isDigest(ref.sourceCommitment) || !isDigest(ref.passageCommitment) || !integer(ref.startByte, 400_000) || !integer(ref.endByte, 400_000) || Number(ref.endByte) <= Number(ref.startByte)) invalid();
    }
    if (check.state !== "completed" && (coverage.checkedUnits || check.sourceRefs.length || calibration.calibrationId) || check.state === "completed" && !coverage.checkedUnits || check.judgment === "meets" && coverage.status !== "complete" || check.judgment === "insufficient_evidence" && check.reasonCode === null || check.kind === "supplied_material" && check.state === "completed" && !check.sourceRefs.length) invalid();
  }
  const { evidenceCommitment, ...payload } = value;
  if (evidenceCommitment !== commitment(payload, domain)) invalid();
  return value as EvaluationEvidence;
}
export function validateEvidenceBinding(input: unknown, result: EvaluationResult, request?: EvaluationRequest): EvaluationEvidence {
  const value = parseEvaluationEvidence(input);
  for (const key of ["workspaceId", "caseId", "modelBundleId", "inputCommitment", "templateCommitment", "resultCommitment", "observedAt"] as const) if (value[key] !== result[key]) invalid();
  const criteria = new Map(result.criteria.map(criterion => [criterion.questionId, criterion]));
  for (const check of value.checks) if (check.state === "completed" && (!check.questionId || !criteria.has(check.questionId) || check.calibration.calibrationId !== criteria.get(check.questionId)!.calibrationId)) invalid();
  if ([...criteria.keys()].some(id => !value.checks.some(check => check.questionId === id))) invalid();
  if (request) {
    const {schemaVersion: _, idempotencyKey: __, deadlineMs: ___, ...committed} = request;
    if (value.inputCommitment !== commitment(committed, "rateloop.evaluator.input.v1") || value.templateCommitment !== commitment(request.template, "rateloop.evaluator.template.v1") || value.language !== request.template.language || value.answerCommitment !== commitment(request.input.text, "rateloop.evaluator.answer.v1")) invalid();
    const questions = new Map(request.template.questions.map(question => [question.id, question]));
    if (value.checks.filter(check => check.questionId !== null).length !== questions.size || [...questions.keys()].some(id => !value.checks.some(check => check.questionId === id))) invalid();
    const sourceRubric = isSuppliedMaterialTemplate(request.template);
    for (const check of value.checks) {
      if (check.state === "completed") {
        if (!["criterion", "supplied_material"].includes(check.kind) || (check.kind === "supplied_material") !== sourceRubric) invalid();
        const question = questions.get(check.questionId!)!;
        const criterion = criteria.get(check.questionId!)!;
        const probabilities = criterion.probabilities ?? criterion.rawScores;
        const highest = Math.max(...Object.values(probabilities));
        if (probabilities[criterion.label] !== highest) invalid();
        const tied = Object.values(probabilities).filter(score => score === highest).length !== 1;
        const expected = !question.passLabels.length || tied || sourceRubric && criterion.label === "insufficient_evidence" ? "insufficient_evidence" : question.passLabels.includes(criterion.label) ? "meets" : "does_not_meet";
        if (check.judgment !== expected || sourceRubric && !request.input.evidence) invalid();
      }
      const bytes = new TextEncoder().encode(request.input.evidence);
      for (const ref of check.sourceRefs) {
        if (ref.sourceId !== "supplied-evidence" || ref.endByte > bytes.length || ref.sourceCommitment !== commitment(request.input.evidence, "rateloop.evaluator.source.v1")) invalid();
        let passage: string; try { passage = new TextDecoder("utf-8", {fatal: true}).decode(bytes.slice(ref.startByte, ref.endByte)); } catch { return invalid(); }
        if (ref.passageCommitment !== commitment(passage, "rateloop.evaluator.passage.v1")) invalid();
      }
    }
  }
  return value;
}
