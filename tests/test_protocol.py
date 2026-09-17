import copy
import json
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError
from rateloop_evaluator.protocol import EvaluationRequest, EvaluationResult, commitment, make_result

ROOT = Path(__file__).resolve().parents[1]


def request():
    return EvaluationRequest.model_validate_json((ROOT / "examples/reply-request.json").read_text())


def test_case_binding_and_retry_identity():
    original = request()
    changed = original.model_copy(update={"idempotencyKey": "another-retry", "deadlineMs": 100})
    assert original.input_commitment() == changed.input_commitment()
    for key in ["workspaceId", "caseId", "modelBundleId"]:
        assert original.input_commitment() != original.model_copy(update={key: "changed"}).input_commitment()


def test_request_rejects_ambiguous_labels_and_extra_fields():
    data = request().model_dump()
    data["template"]["questions"][0]["labels"][1]["id"] = "suitable"
    with pytest.raises(ValidationError): EvaluationRequest.model_validate(data)
    data = request().model_dump(); data["cloudFallback"] = True
    with pytest.raises(ValidationError): EvaluationRequest.model_validate(data)


def test_python_and_typescript_commitment_parity():
    material = {"z": -0.0, "ä": "Grüße", "😀": 0.0000001, "\ufffd": [True, None, 0.3333333333333333]}
    source = "import {commitment} from './contracts/evaluator.ts'; console.log(commitment(JSON.parse(process.argv[1]),'test.v1'));"
    result = subprocess.run(["node", "--experimental-strip-types", "--input-type=module", "-e", source, json.dumps(material)], cwd=ROOT, check=True, capture_output=True, text=True)
    assert result.stdout.strip() == commitment(material, "test.v1")


def test_result_commitment_and_calibration_fail_closed():
    req = request()
    result = make_result(workspaceId=req.workspaceId, caseId=req.caseId, modelBundleId=req.modelBundleId,
                         inputCommitment=req.input_commitment(), templateCommitment=req.template_commitment(),
                         outcome="uncertain", abstainReason="uncalibrated", criteria=[], durationMs=1,
                         observedAt="2026-09-17T12:00:00.000Z")
    data = result.model_dump(); data["outcome"] = "pass"
    with pytest.raises(ValidationError): EvaluationResult.model_validate(data)
    data = result.model_dump(); data["caseId"] = "different"
    with pytest.raises(ValidationError): EvaluationResult.model_validate(data)
    source = "import {parseEvaluationResult} from './contracts/evaluator.ts'; console.log(parseEvaluationResult(JSON.parse(process.argv[1])).resultCommitment);"
    out = subprocess.run(["node", "--experimental-strip-types", "--input-type=module", "-e", source, result.model_dump_json()], cwd=ROOT, check=True, capture_output=True, text=True)
    assert out.stdout.strip() == result.resultCommitment
