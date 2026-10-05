"""Versioned, content-free evidence accompanying an unchanged v1 result.

This envelope records performed checks, not additional model judgments inferred
from a score. Its commitment is integrity, never a claim of human qualification.
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import Field, model_validator

from .protocol import Digest, EvaluationRequest, EvaluationResult, Identifier, WireModel, commitment

EVIDENCE_SCHEMA = "rateloop.evaluator.evidence.v2"


class RuntimeIdentity(WireModel):
    modelId: str | None = Field(default=None, min_length=1, max_length=200)
    modelRevision: str | None = Field(default=None, pattern=r"^[a-f0-9]{40,64}$")
    weightsCommitment: Digest | None = None
    tokenizerCommitment: Digest | None = None
    runtime: str = Field(min_length=1, max_length=160)
    precision: str | None = Field(default=None, min_length=1, max_length=40)
    scoreAdapter: str = Field(min_length=1, max_length=160)


class CheckCoverage(WireModel):
    status: Literal["complete", "partial", "none", "unknown"]
    checkedUnits: int = Field(ge=0, le=100_000)
    totalUnits: int | None = Field(default=None, ge=0, le=100_000)

    @model_validator(mode="after")
    def consistent_counts(self):
        if self.totalUnits is not None and self.checkedUnits > self.totalUnits:
            raise ValueError("Checked units exceed available units")
        if self.status == "complete" and (not self.checkedUnits or self.totalUnits != self.checkedUnits):
            raise ValueError("Complete coverage requires nonzero exact totals")
        if self.status == "none" and self.checkedUnits:
            raise ValueError("Unchecked coverage cannot contain checked units")
        if self.status == "partial" and (not self.checkedUnits or self.totalUnits is not None and self.checkedUnits >= self.totalUnits):
            raise ValueError("Partial coverage must leave units unchecked")
        return self


class SourceReference(WireModel):
    sourceId: Identifier
    sourceCommitment: Digest
    passageCommitment: Digest
    startByte: int = Field(ge=0, le=400_000)
    endByte: int = Field(ge=1, le=400_000)

    @model_validator(mode="after")
    def nonempty_span(self):
        if self.endByte <= self.startByte:
            raise ValueError("Source passage must have a nonempty UTF-8 byte span")
        return self


class CheckCalibration(WireModel):
    # A fitted mapping alone is not independent qualification for public confidence.
    status: Literal["not_validated", "mapping_registered"]
    calibrationId: Identifier | None = None

    @model_validator(mode="after")
    def paired_identity(self):
        if (self.status == "mapping_registered") != (self.calibrationId is not None):
            raise ValueError("Calibration mapping status must match its identity")
        return self


class EvidenceCheck(WireModel):
    id: Identifier
    questionId: Identifier | None = None
    kind: Literal["criterion", "supplied_material", "factual_claims", "code_execution"]
    state: Literal["completed", "not_checked", "not_applicable", "pending", "withheld", "failed"]
    judgment: Literal["meets", "does_not_meet", "insufficient_evidence"] | None = None
    reasonCode: Identifier | None = None
    coverage: CheckCoverage
    sourceRefs: list[SourceReference] = Field(default_factory=list, max_length=20)
    calibration: CheckCalibration

    @model_validator(mode="after")
    def performed_check_only(self):
        if (self.state == "completed") != (self.judgment is not None):
            raise ValueError("Only a completed check has a judgment")
        if self.state != "completed" and (self.coverage.checkedUnits or self.sourceRefs or self.calibration.calibrationId):
            raise ValueError("Unperformed checks cannot carry scored evidence")
        if self.state == "completed" and not self.coverage.checkedUnits:
            raise ValueError("Completed checks require recorded coverage")
        if self.kind == "criterion" and self.questionId is None:
            raise ValueError("A rubric criterion requires its question identity")
        if self.judgment == "meets" and self.coverage.status != "complete":
            raise ValueError("Partial or unknown coverage cannot establish a pass")
        if self.judgment == "insufficient_evidence" and self.reasonCode is None:
            raise ValueError("Abstention requires a reason")
        if self.kind == "supplied_material" and self.state == "completed" and not self.sourceRefs:
            raise ValueError("Source support requires exact inspected material")
        return self


class EvaluationEvidence(WireModel):
    schemaVersion: Literal["rateloop.evaluator.evidence.v2"] = EVIDENCE_SCHEMA
    workspaceId: Identifier
    caseId: Identifier
    modelBundleId: Identifier
    inputCommitment: Digest
    templateCommitment: Digest
    resultCommitment: Digest
    answerCommitment: Digest
    language: Literal["en", "de"]
    observedAt: str
    identity: RuntimeIdentity
    checks: list[EvidenceCheck] = Field(min_length=1, max_length=24)
    evidenceCommitment: Digest

    @model_validator(mode="after")
    def consistent_envelope(self):
        from datetime import datetime
        if not re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z", self.observedAt):
            raise ValueError("Evidence timestamp must use canonical UTC milliseconds")
        datetime.fromisoformat(self.observedAt.replace("Z", "+00:00"))
        if len({check.id for check in self.checks}) != len(self.checks):
            raise ValueError("Duplicate evidence checks")
        if self.evidenceCommitment != commitment(self.model_dump(exclude={"evidenceCommitment"}), EVIDENCE_SCHEMA):
            raise ValueError("Evidence commitment mismatch")
        question_ids = [check.questionId for check in self.checks if check.questionId is not None]
        if len(set(question_ids)) != len(question_ids):
            raise ValueError("A criterion cannot be repeated under multiple checks")
        return self


def make_evidence(**fields) -> EvaluationEvidence:
    value = {"schemaVersion": EVIDENCE_SCHEMA, **fields}
    # Default values are committed explicitly, identically in Python/TypeScript.
    value["identity"] = RuntimeIdentity.model_validate(value["identity"]).model_dump()
    value["checks"] = [EvidenceCheck.model_validate(item).model_dump() for item in value["checks"]]
    value["evidenceCommitment"] = commitment(value, EVIDENCE_SCHEMA)
    return EvaluationEvidence.model_validate(value)


def validate_evidence_binding(evidence: EvaluationEvidence | dict, result: EvaluationResult, request: EvaluationRequest | None = None) -> EvaluationEvidence:
    value = EvaluationEvidence.model_validate(evidence.model_dump() if isinstance(evidence, EvaluationEvidence) else evidence)
    for field in ("workspaceId", "caseId", "modelBundleId", "inputCommitment", "templateCommitment", "resultCommitment", "observedAt"):
        if getattr(value, field) != getattr(result, field):
            raise ValueError("Evidence differs from its exact result")
    scored = {criterion.questionId: criterion for criterion in result.criteria}
    for check in value.checks:
        if check.state == "completed":
            if check.questionId not in scored:
                raise ValueError("Completed evidence requires a performed criterion")
            if check.calibration.calibrationId != scored[check.questionId].calibrationId:
                raise ValueError("Evidence calibration differs from the scored criterion")
    if not set(scored).issubset({check.questionId for check in value.checks}):
        raise ValueError("Evidence cannot omit a performed criterion")
    if request is not None:
        if (request.input_commitment(), request.template_commitment(), request.template.language,
                commitment(request.input.text, "rateloop.evaluator.answer.v1")) != (
                value.inputCommitment, value.templateCommitment, value.language, value.answerCommitment):
            raise ValueError("Evidence differs from the frozen request or answer")
        questions = {question.id: question for question in request.template.questions}
        from .source_evidence import is_supplied_material_template
        source_rubric = is_supplied_material_template(request.template)
        if {check.questionId for check in value.checks if check.questionId is not None} != set(questions):
            raise ValueError("Evidence must cover each requested criterion")
        for check in value.checks:
            if check.state == "completed":
                if check.kind not in ("criterion", "supplied_material"):
                    raise ValueError("No verified factual lookup or code execution is available")
                if (check.kind == "supplied_material") != source_rubric:
                    raise ValueError("Source support requires the exact declared source rubric")
                criterion = scored[check.questionId]
                probabilities = criterion.probabilities or criterion.rawScores
                highest = max(probabilities.values())
                if probabilities[criterion.label] != highest:
                    raise ValueError("Evidence label is not the model's selected maximum")
                tied = sum(score == highest for score in probabilities.values()) != 1
                question = questions[check.questionId]
                expected = "meets" if criterion.label in question.passLabels else "does_not_meet"
                if not question.passLabels or tied or source_rubric and criterion.label == "insufficient_evidence":
                    expected = "insufficient_evidence"
                if check.judgment != expected:
                    raise ValueError("Evidence changes the rubric's actual predicted judgment")
                if source_rubric and not request.input.evidence:
                    raise ValueError("A source judgment requires supplied material")
            for reference in check.sourceRefs:
                validate_source_reference(reference, request.input.evidence)
    return value


def validate_source_reference(reference: SourceReference, material: str) -> None:
    encoded = material.encode("utf-8")
    if reference.sourceId != "supplied-evidence" or reference.endByte > len(encoded):
        raise ValueError("Source reference is outside the supplied material")
    passage = encoded[reference.startByte:reference.endByte].decode("utf-8")
    if (reference.sourceCommitment != commitment(material, "rateloop.evaluator.source.v1")
            or reference.passageCommitment != commitment(passage, "rateloop.evaluator.passage.v1")):
        raise ValueError("Source reference does not match the exact supplied bytes")
