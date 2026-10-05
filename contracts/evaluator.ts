// SPDX-License-Identifier: Apache-2.0
// Portable evaluator v1 contract. Keep parity with protocol.py and golden fixtures.
import { createHash } from "node:crypto";

export type Question = { id: string; text: string; labels: { id: string; description: string }[]; passLabels: string[]; examples?: {text: string; labelId: string}[] };
export type Template = { id: string; version: number; language: "en" | "de"; questions: Question[]; maxTokens: number };
export type EvaluationRequest = {
  schemaVersion: "rateloop.evaluator.request.v1"; workspaceId: string; caseId: string; idempotencyKey: string;
  sourceGroupId: string | null; template: Template; input: { text: string; context: string; evidence: string }; modelBundleId: string; deadlineMs: number;
};
export type Criterion = { questionId: string; label: string; rawScores: Record<string, number> | null;
  probabilities: Record<string, number> | null; calibrationId: string | null };
export type EvaluationResult = {
  schemaVersion: "rateloop.evaluator.result.v1" | "rateloop.evaluator.result.v2"; workspaceId: string; caseId: string; modelBundleId: string;
  inputCommitment: string; templateCommitment: string; outcome: "pass" | "fail" | "uncertain";
  abstainReason: string | null; criteria: Criterion[]; durationMs: number; observedAt: string; resultCommitment: string;
};

const idPattern = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$/u;
const digestPattern = /^sha256:[0-9a-f]{64}$/u;
function fail(message: string): never { throw new Error(`Invalid evaluator contract: ${message}`); }
function string(value: unknown): string {
  if (typeof value !== "string" || /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/u.test(value)) fail("invalid Unicode string");
  return value;
}
function identifier(value: unknown): string { const v = string(value); if (!idPattern.test(v)) fail("identifier"); return v; }
function digest(value: unknown): string { const v = string(value); if (!digestPattern.test(v)) fail("digest"); return v; }
function object(value: unknown, keys?: string[]): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value) || ![Object.prototype, null].includes(Object.getPrototypeOf(value))) fail("object");
  const v = value as Record<string, unknown>;
  if (keys && (Object.keys(v).some(k => !keys.includes(k)) || keys.some(k => !(Object.prototype.hasOwnProperty.call(v, k))))) fail("object fields");
  return v;
}
function scores(value: unknown): Record<string, number> {
  const v = object(value); const keys = Object.keys(v);
  if (keys.length < 2 || keys.length > 12) fail("score labels");
  for (const key of keys) { identifier(key); if (typeof v[key] !== "number" || !Number.isFinite(v[key]) || Number(v[key]) < 0 || Number(v[key]) > 1) fail("probability"); }
  return v as Record<string, number>;
}

/** RFC8785 for JSON values. Object keys use ECMAScript UTF-16 ordering. */
export function canonicalize(value: unknown): string {
  if (value === null) return "null";
  if (typeof value === "string") return JSON.stringify(string(value));
  if (typeof value === "boolean") return JSON.stringify(value);
  if (typeof value === "number") {
    if (!Number.isFinite(value) || (Number.isInteger(value) && !Number.isSafeInteger(value))) fail("non-finite or unsafe number");
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) return `[${value.map(canonicalize).join(",")}]`;
  const v = object(value);
  return `{${Object.keys(v).sort().map(key => `${JSON.stringify(string(key))}:${canonicalize(v[key])}`).join(",")}}`;
}
export function commitment(value: unknown, domain: string): string {
  return `sha256:${createHash("sha256").update(`${domain}\n${canonicalize(value)}`, "utf8").digest("hex")}`;
}
export function verifyEvaluationResultCommitment(value: EvaluationResult): boolean {
  const { resultCommitment, ...payload } = value;
  return resultCommitment === commitment(payload, value.schemaVersion);
}
export function parseEvaluationResult(value: unknown): EvaluationResult {
  const v = object(value, ["schemaVersion", "workspaceId", "caseId", "modelBundleId", "inputCommitment", "templateCommitment", "outcome", "abstainReason", "criteria", "durationMs", "observedAt", "resultCommitment"]);
  if (!["rateloop.evaluator.result.v1", "rateloop.evaluator.result.v2"].includes(String(v.schemaVersion))) fail("version");
  const labelOnly = v.schemaVersion === "rateloop.evaluator.result.v2";
  for (const k of ["workspaceId", "caseId", "modelBundleId"]) identifier(v[k]);
  for (const k of ["inputCommitment", "templateCommitment", "resultCommitment"]) digest(v[k]);
  if (!["pass", "fail", "uncertain"].includes(String(v.outcome))) fail("outcome");
  if (v.abstainReason !== null && (typeof v.abstainReason !== "string" || v.abstainReason.length > 160)) fail("abstention");
  if (!Number.isSafeInteger(v.durationMs) || Number(v.durationMs) < 0 || Number(v.durationMs) > 2**31-1) fail("duration");
  const observedAt = string(v.observedAt);
  if (!/^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z$/u.test(observedAt) || !Number.isFinite(Date.parse(observedAt)) || new Date(observedAt).toISOString() !== observedAt) fail("timestamp");
  if (!Array.isArray(v.criteria) || v.criteria.length > 20) fail("criteria");
  const questionIds = new Set<string>();
  for (const item of v.criteria) {
    const c = object(item, ["questionId", "label", "rawScores", "probabilities", "calibrationId"]);
    const q = identifier(c.questionId); const label = identifier(c.label);
    if (questionIds.has(q)) fail("duplicate question"); questionIds.add(q);
    if (labelOnly) {
      if (c.rawScores !== null || c.probabilities !== null || c.calibrationId !== null) fail("label-only scores");
      continue;
    }
    const raw = scores(c.rawScores); if (!Object.hasOwn(raw, label)) fail("predicted label");
    if ((c.probabilities === null) !== (c.calibrationId === null)) fail("calibration binding");
    if (c.probabilities !== null) {
      identifier(c.calibrationId); const p = scores(c.probabilities);
      if (Object.keys(p).sort().join("\0") !== Object.keys(raw).sort().join("\0") || Math.abs(Object.values(p).reduce((a,b) => a+b, 0)-1) > 1e-6) fail("distribution");
    }
  }
  if (labelOnly && (v.outcome !== "uncertain" || typeof v.abstainReason !== "string" || !v.abstainReason)) fail("label-only advisory result");
  if (v.outcome !== "uncertain" && (v.abstainReason !== null || !v.criteria.length || v.criteria.some(c => c.probabilities === null))) fail("uncalibrated decision");
  const result = v as EvaluationResult;
  if (!verifyEvaluationResultCommitment(result)) fail("result commitment mismatch");
  return result;
}

/** One judgment mapping for score-based and explicitly label-only adapters. */
export function evaluatorCriterionJudgment(criterion: Criterion, question: Question, source = false): "meets" | "does_not_meet" | "insufficient_evidence" {
  if (!question.labels.some(item => item.id === criterion.label)) fail("predicted label outside rubric");
  const values = criterion.probabilities ?? criterion.rawScores;
  if (values !== null) {
    const highest = Math.max(...Object.values(values));
    if (values[criterion.label] !== highest) fail("label is not score maximum");
    if (Object.values(values).filter(value => value === highest).length !== 1) return "insufficient_evidence";
  }
  if (!question.passLabels.length || (source && criterion.label === "insufficient_evidence")) return "insufficient_evidence";
  return question.passLabels.includes(criterion.label) ? "meets" : "does_not_meet";
}
