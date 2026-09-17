"""Explicit opt-in hardware checks using authored synthetic fixtures, never customer data.

RATELOOP_TEST_MODEL_DIR=/path/to/provisioned/model RATELOOP_TEST_DEVICE=mps pytest tests/test_training_real.py
These labels exercise software mechanics; they are not evidence of human-rating quality.
"""
import os
import json
from argparse import Namespace
import socket
import time

from cryptography.fernet import Fernet
import pytest

from rateloop_evaluator import cli
from rateloop_evaluator.learning import LearningStore
from rateloop_evaluator.registry import provision_signing_key
from rateloop_evaluator.protocol import EvaluationRequest
from rateloop_evaluator.training import TrainOptions, train_snapshot

pytestmark = pytest.mark.skipif(not os.environ.get("RATELOOP_TEST_MODEL_DIR"),
                                reason="Explicit provisioned model and local training opt-in required")


@pytest.mark.parametrize("method", ["full", "lora"])
def test_real_train_optimizer_save_reload_and_revocation(tmp_path, monkeypatch, capsys, method):
    def reject(*args, **kwargs):
        raise AssertionError("Offline training attempted network access")
    monkeypatch.setattr(socket.socket, "connect", reject)
    key = tmp_path / "store.key"
    key.write_bytes(Fernet.generate_key())
    key.chmod(0o600)
    store = LearningStore(tmp_path / "learning", key)
    grant = store.add_grant(workspace_id="synthetic", rights=["ai_use", "private_training"],
                            expires_at=time.time() + 3600, evidence="Authored synthetic hardware fixture")
    template = {"id": "politeness", "version": 1, "language": "en", "maxTokens": 512,
                "questions": [{"id": "tone", "text": "Is the customer reply polite?", "labels": [
                    {"id": "yes", "description": "Polite and courteous"},
                    {"id": "no", "description": "Rude or hostile"}], "passLabels": ["yes"]}]}
    fixtures = [("Thank you for contacting support. We will help.", "yes"),
                ("We appreciate your patience and will investigate.", "yes"),
                ("Please send your order number so we can assist.", "yes"),
                ("Stop wasting our time with stupid questions.", "no"),
                ("Go away. Nobody cares about your problem.", "no"),
                ("This is your fault. Do it yourself.", "no")]
    for index, (text, label) in enumerate(fixtures):
        request = EvaluationRequest(workspaceId="synthetic", caseId=f"case{index}",
            idempotencyKey=f"fixture-{index}", template=template,
            input={"text": text, "context": "Customer asks about a late delivery", "evidence": ""},
            modelBundleId="base")
        store.record_evaluation(evaluation_id=f"eval{index}", workspace_id="synthetic", case_id=request.caseId,
            input_commitment=request.input_commitment(), template_commitment=request.template_commitment(),
            template=template, input_payload=request.input.model_dump())
        store.add_feedback(workspace_id="synthetic", evaluation_id=f"eval{index}",
            input_commitment=request.input_commitment(), template_commitment=request.template_commitment(),
            annotator_id="synthetic-fixture-author", labels={"tone": label}, exposed_to_ai=False,
            independent_human=True)
    snapshot = store.create_snapshot("synthetic", "politeness", 1)
    result = train_snapshot(store, snapshot["id"], "synthetic", os.environ["RATELOOP_TEST_MODEL_DIR"],
        tmp_path / "training", bundle_id=f"smoke-{method}", options=TrainOptions(method=method,
        device=os.environ.get("RATELOOP_TEST_DEVICE", "cpu"), max_steps=1, epochs=1))
    assert result["training"]["optimizerSteps"] == 1
    assert result["training"]["trainableParametersChanged"] is True
    assert result["training"]["reloadMaxAbsoluteError"] < 1e-4
    assert set(result["training"]["trainingExampleIds"]) == {row["evaluation_id"] for row in snapshot["train"]}
    assert not set(result["training"]["trainingExampleIds"]) & {
        row["evaluation_id"] for row in snapshot["calibration"] + snapshot["test"]}
    store.assert_model_usable(f"smoke-{method}", "synthetic")
    if method == "lora":
        # Use the checkpoint from the optimizer test; do not train a second model.
        signing_key = tmp_path / "signing.key"
        provision_signing_key(signing_key)
        cli.write_private(tmp_path / "config.json", {"workspaceId": "synthetic",
            "encryptionKey": str(key), "signingKey": str(signing_key), "tokens": []})
        capsys.readouterr()
        def command(*arguments, expected=0):
            code = cli.main(["--state-dir", str(tmp_path), *map(str, arguments)])
            output = capsys.readouterr()
            assert code == expected, output.err
            return output.err if expected else json.loads(output.out)
        calibration_path = tmp_path / "calibrations.json"
        command("calibrate", "--snapshot-id", snapshot["id"], "--model-dir", result["modelDir"],
                "--bundle-id", "smoke-lora", "--output", calibration_path,
                "--device", os.environ.get("RATELOOP_TEST_DEVICE", "cpu"))
        calibrations = json.loads(calibration_path.read_text())
        assert all(item["model_weights_sha256"] == result["manifest"]["files"]["model.safetensors"]
                   for item in calibrations)
        assert set(calibrations[0]["example_ids"]) == {row["group_id"] for row in snapshot["calibration"]}
        request_path = tmp_path / "request.json"
        cli.write_private(request_path, request.model_copy(update={"modelBundleId": "smoke-lora"}).model_dump())
        policy_path = tmp_path / "selective-policy.json"
        cli.write_private(policy_path, {"threshold": .95, "max_false_approval_rate": .01,
                                        "minimum_coverage": .3, "confidence": .95})
        registered = command("register", "--model-dir", result["modelDir"], "--request", request_path,
            "--snapshot-id", snapshot["id"], "--calibrations", calibration_path, "--selective-policy", policy_path)
        assert registered["mode"] == "shadow"
        evidence_path = tmp_path / "test-evidence.json"
        scored = command("score-test", "--bundle-id", "smoke-lora", "--output", evidence_path,
                         "--device", os.environ.get("RATELOOP_TEST_DEVICE", "cpu"))
        assert scored["groups"] == len({row["group_id"] for row in snapshot["test"]})
        evidence = json.loads(evidence_path.read_text())
        assert evidence["synthetic"] is True
        assert {row["evaluation_id"] for row in evidence["rows"]} == {
            row["evaluation_id"] for row in snapshot["test"]}
        rejected = command("promote", "--bundle-id", "smoke-lora", "--template-commitment", request.template_commitment(),
            "--language", "en", "--mode", "selective", "--evidence", evidence_path, expected=1)
        assert "Synthetic" in rejected
        _, _, _, registry = cli.state(Namespace(state_dir=str(tmp_path)))
        assert registry.active("synthetic", request.template_commitment(), "en")["mode"] == "shadow"
    store.revoke_grant(grant["id"], "synthetic")
    with pytest.raises(PermissionError):
        store.assert_model_usable(f"smoke-{method}", "synthetic")
