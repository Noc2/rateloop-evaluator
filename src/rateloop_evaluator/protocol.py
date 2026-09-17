"""Strict portable wire models and RFC 8785 commitments. No network or model imports."""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Annotated, Literal

import rfc8785
from pydantic import BaseModel, ConfigDict, Field, model_validator

Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$")]
Digest = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
Probability = Annotated[float, Field(ge=0, le=1)]


class WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)


class Label(WireModel):
    id: Identifier
    description: str = Field(min_length=1, max_length=500)


class Question(WireModel):
    id: Identifier
    text: str = Field(min_length=1, max_length=1000)
    labels: list[Label] = Field(min_length=2, max_length=12)
    passLabels: list[Identifier] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def valid_labels(self):
        ids = [label.id for label in self.labels]
        if len(set(ids)) != len(ids) or len(set(self.passLabels)) != len(self.passLabels):
            raise ValueError("Label IDs must be unique")
        if not set(self.passLabels).issubset(ids):
            raise ValueError("passLabels must belong to the declared labels")
        return self


class Template(WireModel):
    id: Identifier
    version: int = Field(ge=1, le=2**31-1)
    language: Literal["en", "de"]
    questions: list[Question] = Field(min_length=1, max_length=20)
    maxTokens: int = Field(default=512, ge=32, le=8192)

    @model_validator(mode="after")
    def unique_questions(self):
        if len({q.id for q in self.questions}) != len(self.questions):
            raise ValueError("Question IDs must be unique")
        return self


class CaseInput(WireModel):
    text: str = Field(min_length=1, max_length=100_000)
    context: str = Field(default="", max_length=100_000)
    evidence: str = Field(default="", max_length=100_000)

    def render(self) -> str:
        # These boundaries are data structure, never model instructions.
        return f"REQUEST/CONTEXT:\n{self.context}\n\nEVIDENCE:\n{self.evidence}\n\nREPLY/ARTIFACT:\n{self.text}"


class EvaluationRequest(WireModel):
    schemaVersion: Literal["rateloop.evaluator.request.v1"] = "rateloop.evaluator.request.v1"
    workspaceId: Identifier
    caseId: Identifier
    sourceGroupId: Identifier | None = None
    idempotencyKey: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,199}$")
    template: Template
    input: CaseInput
    modelBundleId: Identifier
    deadlineMs: int = Field(default=5000, ge=1, le=60_000)

    def committed_input(self) -> dict:
        return self.model_dump(exclude={"schemaVersion", "idempotencyKey", "deadlineMs"})

    def input_commitment(self) -> str:
        return commitment(self.committed_input(), "rateloop.evaluator.input.v1")

    def template_commitment(self) -> str:
        return commitment(self.template.model_dump(), "rateloop.evaluator.template.v1")


class Criterion(WireModel):
    questionId: Identifier
    label: Identifier
    rawScores: dict[Identifier, Probability]
    probabilities: dict[Identifier, Probability] | None = None
    calibrationId: Identifier | None = None

    @model_validator(mode="after")
    def valid_distribution(self):
        if not 2 <= len(self.rawScores) <= 12 or self.label not in self.rawScores:
            raise ValueError("Incomplete score labels")
        if (self.probabilities is None) != (self.calibrationId is None):
            raise ValueError("Calibration identity and probabilities must be paired")
        if self.probabilities is not None:
            if set(self.probabilities) != set(self.rawScores) or abs(sum(self.probabilities.values()) - 1) > 1e-6:
                raise ValueError("Invalid calibrated distribution")
        return self


class EvaluationResult(WireModel):
    schemaVersion: Literal["rateloop.evaluator.result.v1"] = "rateloop.evaluator.result.v1"
    workspaceId: Identifier
    caseId: Identifier
    modelBundleId: Identifier
    inputCommitment: Digest
    templateCommitment: Digest
    outcome: Literal["pass", "fail", "uncertain"]
    abstainReason: str | None = Field(default=None, max_length=160)
    criteria: list[Criterion] = Field(max_length=20)
    durationMs: int = Field(ge=0, le=2**31-1)
    observedAt: str
    resultCommitment: Digest

    @model_validator(mode="after")
    def result_invariants(self):
        if not re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z", self.observedAt):
            raise ValueError("observedAt must be canonical ISO milliseconds UTC")
        datetime.fromisoformat(self.observedAt.replace("Z", "+00:00"))
        if len({item.questionId for item in self.criteria}) != len(self.criteria):
            raise ValueError("Duplicate question results")
        if self.outcome != "uncertain" and (self.abstainReason is not None or not self.criteria or any(c.probabilities is None for c in self.criteria)):
            raise ValueError("A decision requires calibrated criteria and no abstention")
        expected = commitment(self.model_dump(exclude={"resultCommitment"}), "rateloop.evaluator.result.v1")
        if self.resultCommitment != expected:
            raise ValueError("Result commitment mismatch")
        return self


def commitment(value: object, domain: str) -> str:
    return "sha256:" + hashlib.sha256(domain.encode("utf-8") + b"\n" + rfc8785.dumps(value)).hexdigest()


def make_result(**fields) -> EvaluationResult:
    value = {"schemaVersion": "rateloop.evaluator.result.v1", "abstainReason": None, **fields}
    value["resultCommitment"] = commitment(value, "rateloop.evaluator.result.v1")
    return EvaluationResult.model_validate(value)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
