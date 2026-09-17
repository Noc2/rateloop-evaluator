// SPDX-License-Identifier: Apache-2.0
import { commitment, parseEvaluationResult } from "../../contracts/evaluator.ts";
import type { EvaluationRequest, EvaluationResult } from "../../contracts/evaluator.ts";
export type { EvaluationRequest, EvaluationResult } from "../../contracts/evaluator.ts";

/** Server-side client. Keep the local token out of browser bundles. */
export class EvaluatorClient {
  private endpoint: URL;
  private token: string;
  private send: typeof fetch;
  constructor(options: { endpoint: string; token: string; fetch?: typeof fetch }) {
    const url = new URL(options.endpoint);
    if (url.username || url.password || url.search || url.hash || url.pathname !== "/") throw new Error("Use an evaluator origin without credentials, path or query");
    if (url.protocol !== "https:" && !(url.protocol === "http:" && ["localhost", "127.0.0.1", "[::1]"].includes(url.hostname))) throw new Error("Remote evaluators require HTTPS");
    if (!options.token || /[\r\n]/u.test(options.token)) throw new Error("A scoped local token is required");
    this.endpoint = url; this.token = options.token; this.send = options.fetch ?? fetch;
  }
  async evaluate(request: EvaluationRequest): Promise<EvaluationResult> {
    const response = await this.send(new URL("/v1/evaluate", this.endpoint), {
      method: "POST", redirect: "error", signal: AbortSignal.timeout(Math.min(request.deadlineMs + 5000, 65000)),
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${this.token}` }, body: JSON.stringify(request),
    });
    if (!response.ok) { await response.body?.cancel(); throw new Error(`Local evaluator returned ${response.status}; human review remains required`); }
    if (!response.body) throw new Error("Missing evaluator response");
    const reader = response.body.getReader(); const chunks: Uint8Array[] = []; let length = 0;
    while (true) {
      const item = await reader.read(); if (item.done) break;
      length += item.value.byteLength;
      if (length > 100_000) { await reader.cancel(); throw new Error("Evaluator response exceeds limit"); }
      chunks.push(item.value);
    }
    const bytes = new Uint8Array(length); let offset = 0;
    for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
    const result = parseEvaluationResult(JSON.parse(new TextDecoder("utf-8", {fatal:true}).decode(bytes)));
    const { schemaVersion: _schema, idempotencyKey: _retry, deadlineMs: _deadline, ...input } = request;
    if (result.workspaceId !== request.workspaceId || result.caseId !== request.caseId || result.modelBundleId !== request.modelBundleId
      || result.inputCommitment !== commitment(input, "rateloop.evaluator.input.v1")
      || result.templateCommitment !== commitment(request.template, "rateloop.evaluator.template.v1")) throw new Error("Evaluator response belongs to a different case or template");
    const questions = new Map(request.template.questions.map(question => [question.id, question]));
    if (result.criteria.length && result.criteria.length !== questions.size) throw new Error("Evaluator returned incomplete criteria");
    for (const criterion of result.criteria) {
      const question = questions.get(criterion.questionId);
      if (!question || question.labels.map(label => label.id).sort().join("\0") !== Object.keys(criterion.rawScores).sort().join("\0")) throw new Error("Evaluator labels differ from the request");
    }
    return result;
  }
}
