"""Numerical calibration and exact one-sided binomial error bounds.

A temperature fit is not a correctness certificate. The registry separately requires
independent held-out deployment evidence at the selected operating threshold.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Scores must be finite numbers")
    return float(value)


def _logits(scores: dict[str, float], score_type: str) -> dict[str, float]:
    if not isinstance(scores, dict) or len(scores) < 2 or any(not isinstance(k, str) or not k for k in scores):
        raise ValueError("At least two labeled scores are required")
    values = {k: _number(v) for k, v in scores.items()}
    if score_type == "probabilities":
        if any(v < 0 or v > 1 for v in values.values()) or sum(values.values()) <= 0:
            raise ValueError("Probability scores must lie in [0,1] and have positive mass")
        return {k: math.log(max(v, 1e-12)) for k, v in values.items()}
    if score_type != "logits":
        raise ValueError("Unsupported score type")
    return values


def _softmax(logits: dict[str, float], temperature: float) -> dict[str, float]:
    offset = max(logits.values())
    values = {k: math.exp((v - offset) / temperature) for k, v in logits.items()}
    denominator = sum(values.values())
    return {k: v / denominator for k, v in values.items()}


def fit_temperature(
    scores: list[dict[str, float]], labels: list[str], *, model_bundle_id: str,
    template_commitment: str, question_id: str, language: str,
    example_ids: list[str], score_type: str = "probabilities",
) -> dict[str, Any]:
    """Fit a scalar temperature on calibration-only human labels.

    IDs identify independent groups, not repeated annotations of the same input.
    The caller must keep these groups out of training and final test evidence.
    """
    if not scores or len(scores) != len(labels) or len(scores) != len(example_ids):
        raise ValueError("Scores, labels and example IDs must be nonempty and aligned")
    if len(set(example_ids)) != len(example_ids) or any(not x for x in example_ids):
        raise ValueError("Calibration example IDs must be unique")
    if any(not value for value in (model_bundle_id, template_commitment, question_id, language)):
        raise ValueError("Calibration must bind a bundle, template, question and language")
    logits = [_logits(row, score_type) for row in scores]
    keys = set(logits[0])
    if any(set(row) != keys or label not in keys for row, label in zip(logits, labels)):
        raise ValueError("Calibration label sets must match exactly")
    # Convex in inverse temperature. Minimize NLL in log-temperature over a
    # deliberately bounded domain; no scipy/GPU dependency is necessary.
    def loss(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        return sum(-math.log(max(_softmax(row, temperature)[label], 1e-300)) for row, label in zip(logits, labels)) / len(labels)
    left, right = math.log(.05), math.log(20)
    for _ in range(100):
        a, b = left + (right-left)/3, right - (right-left)/3
        if loss(a) <= loss(b):
            right = b
        else:
            left = a
    temperature = math.exp((left+right)/2)
    artifact = {
        "schema_version": "rateloop.calibration.v1", "model_bundle_id": model_bundle_id,
        "template_commitment": template_commitment, "question_id": question_id,
        "language": language, "label_ids": sorted(keys), "temperature": temperature,
        "score_type": score_type, "example_ids": list(example_ids), "sample_count": len(labels),
        "nll_before": loss(0), "nll_after": loss(math.log(temperature)),
    }
    artifact["id"] = "cal_" + hashlib.sha256(json.dumps(artifact, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return artifact


def validate_calibration(artifact: dict[str, Any]) -> None:
    candidate = dict(artifact)
    artifact_id = candidate.pop("id", None)
    expected = "cal_" + hashlib.sha256(json.dumps(candidate, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if artifact_id != expected or candidate.get("schema_version") != "rateloop.calibration.v1":
        raise ValueError("Calibration digest or schema is invalid")
    t = _number(candidate.get("temperature"))
    if not .05 <= t <= 20 or len(set(candidate.get("label_ids", []))) < 2:
        raise ValueError("Calibration parameters are invalid")
    ids = candidate.get("example_ids", [])
    if not ids or len(ids) != len(set(ids)) or candidate.get("sample_count") != len(ids):
        raise ValueError("Calibration evidence identifiers are invalid")
    for field in ("model_bundle_id", "template_commitment", "question_id", "language"):
        if not isinstance(candidate.get(field), str) or not candidate[field]:
            raise ValueError("Calibration binding is missing")
    if candidate.get("score_type") not in ("probabilities", "logits"):
        raise ValueError("Calibration score type is invalid")
    for field in ("nll_before", "nll_after"):
        if _number(candidate.get(field)) < 0:
            raise ValueError("Calibration loss is invalid")


def apply_temperature(scores: dict[str, float], artifact: dict[str, Any], *, model_bundle_id: str,
                      template_commitment: str, question_id: str, language: str) -> dict[str, float]:
    validate_calibration(artifact)
    for key, expected in {"model_bundle_id": model_bundle_id, "template_commitment": template_commitment,
                          "question_id": question_id, "language": language}.items():
        if artifact[key] != expected:
            raise ValueError(f"Calibration {key} mismatch")
    if sorted(scores) != artifact["label_ids"]:
        raise ValueError("Calibration label set mismatch")
    return _softmax(_logits(scores, artifact["score_type"]), artifact["temperature"])


def false_approval_upper_bound(errors: int, total: int, confidence: float = .95) -> float:
    """Exact one-sided Clopper-Pearson bound for independent auto-approved cases.

    `total` counts audited auto-approvals, not all reviewed cases. Correlated cases
    must be grouped upstream; zero observed errors is not zero population risk.
    """
    if isinstance(errors, bool) or isinstance(total, bool) or not isinstance(errors, int) or not isinstance(total, int):
        raise ValueError("Counts must be integers")
    if total <= 0 or not 0 <= errors <= total or not 0 < confidence < 1:
        raise ValueError("Invalid binomial evidence")
    if errors == total:
        return 1.0
    if errors == 0:
        return -math.expm1(math.log1p(-confidence) / total)
    log_coefficients = [math.lgamma(total+1) - math.lgamma(i+1) - math.lgamma(total-i+1) for i in range(errors+1)]
    def cdf(p: float) -> float:
        terms = [log_coefficients[i] + i*math.log(p) + (total-i)*math.log1p(-p) for i in range(errors+1)]
        maximum = max(terms)
        return math.exp(maximum) * sum(math.exp(t-maximum) for t in terms)
    left, right = errors/total, 1.0
    target = 1-confidence
    for _ in range(100):
        mid = (left+right)/2
        if mid == left or mid == right:
            break
        if cdf(mid) > target:
            left = mid
        else:
            right = mid
    return (left+right)/2
