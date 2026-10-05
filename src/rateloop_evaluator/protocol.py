"""Strict portable wire models and RFC 8785 commitments. No network or model imports."""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Annotated, Iterable, Literal

import rfc8785
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_serializer, model_validator

Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$")]
Digest = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
Probability = Annotated[float, Field(ge=0, le=1)]


class WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)


class Label(WireModel):
    id: Identifier
    description: str = Field(min_length=1, max_length=500)


class Demonstration(WireModel):
    text: str = Field(min_length=1, max_length=600)
    labelId: Identifier

    @field_validator("text", mode="before")
    @classmethod
    def canonical_text(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("Demonstration text must be a string")
        # ECMAScript trim and UTF-16 lengths match the portable website contract.
        value = value.strip("\u0009\u000a\u000b\u000c\u000d\u0020\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff")
        if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u061c\u200e\u200f\u202a-\u202e\u2066-\u2069\ud800-\udfff]", value):
            raise ValueError("Demonstrations must contain valid visible text")
        if not 1 <= len(value.encode("utf-16-le")) // 2 <= 600:
            raise ValueError("Demonstration text must contain 1–600 UTF-16 units")
        return value


class Question(WireModel):
    id: Identifier
    text: str = Field(min_length=1, max_length=1000)
    labels: list[Label] = Field(min_length=2, max_length=12)
    passLabels: list[Identifier] = Field(default_factory=list, max_length=12)
    examples: list[Demonstration] = Field(default_factory=list, max_length=4)

    @model_serializer(mode="wrap")
    def omit_empty_examples(self, handler):
        value = handler(self)
        if not self.examples:
            value.pop("examples", None)
        return value

    @model_validator(mode="after")
    def valid_labels(self):
        ids = [label.id for label in self.labels]
        if len(set(ids)) != len(ids) or len(set(self.passLabels)) != len(self.passLabels):
            raise ValueError("Label IDs must be unique")
        if not set(self.passLabels).issubset(ids):
            raise ValueError("passLabels must belong to the declared labels")
        if any(example.labelId not in ids for example in self.examples):
            raise ValueError("Demonstration labels must belong to the declared labels")
        if sum(len(example.text.encode("utf-16-le")) // 2 for example in self.examples) > 1600:
            raise ValueError("Demonstrations must not exceed 1600 UTF-16 units in total")
        return self


def validate_no_demonstration_overlap(input_text: str, questions: list[dict]) -> None:
    """Do not count a rubric's disclosed answer as separate evaluation evidence.

    ECMAScript whitespace collapse is shared with the website importer. Context
    does not make a duplicate artifact independent. This catches exact normalized
    text, not paraphrases or other near-duplicate leakage.
    """
    def normalized(value: str) -> str:
        return re.sub(r"[\u0009-\u000d\u0020\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff]+", " ", value).strip(" ")
    text = normalized(input_text)
    if any(text == normalized(example.text)
           for question in questions for example in Question.model_validate(question).examples):
        raise ValueError("Dataset rows must not repeat a rubric demonstration")


def validate_reference_demonstration_isolation(rows: Iterable[dict]) -> None:
    """Blind human provenance cannot make a disclosed prompt answer held out.

    Check every row, including nonrepresentative members of a source group, at
    snapshot and qualification boundaries. Older templates with no examples
    retain their existing validation and serialization behavior.
    """
    for row in rows:
        questions = row.get("template", {}).get("questions", [])
        if any(question.get("examples") for question in questions):
            payload = row.get("input")
            if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
                raise ValueError("Reference rows with rubric demonstrations require retained input text")
            validate_no_demonstration_overlap(payload["text"], questions)


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
    rawScores: dict[Identifier, Probability] | None
    probabilities: dict[Identifier, Probability] | None = None
    calibrationId: Identifier | None = None

    @model_validator(mode="after")
    def valid_distribution(self):
        if self.rawScores is None:
            if self.probabilities is not None or self.calibrationId is not None:
                raise ValueError("Label-only judgments cannot claim numeric calibration")
            return self
        if not 2 <= len(self.rawScores) <= 12 or self.label not in self.rawScores:
            raise ValueError("Incomplete score labels")
        if (self.probabilities is None) != (self.calibrationId is None):
            raise ValueError("Calibration identity and probabilities must be paired")
        if self.probabilities is not None:
            if set(self.probabilities) != set(self.rawScores) or abs(sum(self.probabilities.values()) - 1) > 1e-6:
                raise ValueError("Invalid calibrated distribution")
        return self


class EvaluationResult(WireModel):
    model_config = ConfigDict(json_schema_extra={"allOf": [{
        "if": {"properties": {"schemaVersion": {"const": "rateloop.evaluator.result.v1"}}},
        "then": {"properties": {"criteria": {"items": {"properties": {"rawScores": {"type": "object"}}}}}},
        "else": {"properties": {"outcome": {"const": "uncertain"}, "abstainReason": {"type": "string", "minLength": 1},
            "criteria": {"items": {"properties": {"rawScores": {"type": "null"}, "probabilities": {"type": "null"}, "calibrationId": {"type": "null"}}}}}}
    }]})
    schemaVersion: Literal["rateloop.evaluator.result.v1", "rateloop.evaluator.result.v2"] = "rateloop.evaluator.result.v1"
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
        if self.schemaVersion == "rateloop.evaluator.result.v1" and any(c.rawScores is None for c in self.criteria):
            raise ValueError("V1 requires complete raw score vectors")
        if self.schemaVersion == "rateloop.evaluator.result.v2" and (
                any(c.rawScores is not None or c.probabilities is not None or c.calibrationId is not None for c in self.criteria)
                or self.outcome != "uncertain" or not self.abstainReason):
            raise ValueError("V2 label-only judgments must remain uncalibrated advisory results")
        if not re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z", self.observedAt):
            raise ValueError("observedAt must be canonical ISO milliseconds UTC")
        datetime.fromisoformat(self.observedAt.replace("Z", "+00:00"))
        if len({item.questionId for item in self.criteria}) != len(self.criteria):
            raise ValueError("Duplicate question results")
        if self.outcome != "uncertain" and (self.abstainReason is not None or not self.criteria or any(c.probabilities is None for c in self.criteria)):
            raise ValueError("A decision requires calibrated criteria and no abstention")
        expected = commitment(self.model_dump(exclude={"resultCommitment"}), self.schemaVersion)
        if self.resultCommitment != expected:
            raise ValueError("Result commitment mismatch")
        return self


def commitment(value: object, domain: str) -> str:
    return "sha256:" + hashlib.sha256(domain.encode("utf-8") + b"\n" + rfc8785.dumps(value)).hexdigest()


def make_result(**fields) -> EvaluationResult:
    value = {"schemaVersion": "rateloop.evaluator.result.v1", "abstainReason": None, **fields}
    value["resultCommitment"] = commitment(value, value["schemaVersion"])
    return EvaluationResult.model_validate(value)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def criterion_judgment(criterion: Criterion, question: Question, source: bool = False) -> tuple[str, str | None]:
    """One judgment mapping for scored and explicitly label-only adapters."""
    if criterion.label not in {label.id for label in question.labels}:
        raise ValueError("Predicted label is outside the rubric")
    scores = criterion.probabilities or criterion.rawScores
    if scores is not None:
        maximum = max(scores.values())
        if scores[criterion.label] != maximum: raise ValueError("Label is not the score maximum")
        if sum(value == maximum for value in scores.values()) != 1: return "insufficient_evidence", "tied_scores"
    if not question.passLabels: return "insufficient_evidence", "rubric_has_no_judgment_mapping"
    if source and criterion.label == "insufficient_evidence": return "insufficient_evidence", "insufficient_source_evidence"
    return ("meets" if criterion.label in question.passLabels else "does_not_meet"), None
