"""CLI-to-service lifecycle checks. Only neural computation is replaced in tests."""
import json
from pathlib import Path
import stat

from fastapi.testclient import TestClient
import pytest

from rateloop_evaluator import cli
from rateloop_evaluator.backends import MODEL_FILES, MODEL_ID, MODEL_REVISION, write_model_manifest
from rateloop_evaluator.protocol import EvaluationRequest

ROOT = Path(__file__).resolve().parents[1]


def invoke(capsys, state, *arguments, expected=0):
    code = cli.main(["--state-dir", str(state), *map(str, arguments)])
    captured = capsys.readouterr()
    assert code == expected, captured.err
    if expected:
        return captured.err
    return json.loads(captured.out) if captured.out.strip() else None


def request_file(tmp_path, bundle="base-test"):
    request = json.loads((ROOT / "examples/reply-request.json").read_text())
    request.update(workspaceId="workspace-test", modelBundleId=bundle)
    path = tmp_path / (bundle + "-request.json")
    path.write_text(json.dumps(request))
    return path, request


def model_files(tmp_path, training=None):
    directory = tmp_path / "model"
    directory.mkdir()
    for relative in MODEL_FILES[:-1]:
        path = directory / relative
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({"architecture": "boundary", "max_len": 4096})
                        if path.suffix == ".json" else "synthetic-test-weights")
    write_model_manifest(directory, source={"repository": MODEL_ID, "revision": MODEL_REVISION,
                                           "license": "Apache-2.0"}, training=training)
    return directory


@pytest.fixture
def initialized(tmp_path, capsys):
    state = tmp_path / "state"
    invoke(capsys, state, "init", "--workspace", "workspace-test")
    return state


def test_init_accepts_existing_empty_directory_and_does_not_grant_training(tmp_path, capsys):
    state = tmp_path / "empty"
    state.mkdir()
    report = invoke(capsys, state, "init", "--workspace", "workspace-test")
    assert report["grants"] == []
    config = json.loads((state / "config.json").read_text())
    client = json.loads((state / "client.json").read_text())
    assert config["tokens"][0]["roles"] == ["evaluate"]
    assert client["token"] not in json.dumps(report)
    for name in ("config.json", "client.json", "encryption.key", "signing.key"):
        assert stat.S_IMODE((state / name).stat().st_mode) == 0o600
    invoke(capsys, state, "init", "--workspace", "workspace-test", expected=1)


def test_cli_register_serve_review_snapshot_train_and_revoke(initialized, tmp_path, capsys, monkeypatch):
    state = initialized
    model_dir = model_files(tmp_path)
    request_path, body = request_file(tmp_path)
    template_id = body["template"]["id"]
    private = invoke(capsys, state, "grant", "--right", "private_training", "--template", template_id,
                     "--evidence", "Authored synthetic test fixture")
    invoke(capsys, state, "grant", "--right", "ai_use", "--template", template_id,
           "--evidence", "Authored synthetic test fixture")
    reviewer_path = tmp_path / "reviewer.json"
    invoke(capsys, state, "issue-reviewer-token", "--reviewer-id", "reviewer-test", "--output", reviewer_path)
    registered = invoke(capsys, state, "register", "--model-dir", model_dir, "--request", request_path)
    assert registered["mode"] == "shadow"

    class NeuralTestDouble:
        def __init__(self, model_dir, device):
            self.model_dir, self.device = model_dir, device
        def load(self): return self
        def count_tokens(self, text, questions): return 80
        def predict(self, text, questions):
            return {q["id"]: {label["id"]: 1 / len(q["labels"]) for label in q["labels"]} for q in questions}

    import rateloop_evaluator.backends
    import uvicorn
    served = []
    monkeypatch.setattr(rateloop_evaluator.backends, "GLiNERBackend", NeuralTestDouble)
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: served.append((app, kwargs)))
    invoke(capsys, state, "serve", "--bundle-id", body["modelBundleId"])
    assert served[0][1]["host"] == "127.0.0.1" and served[0][1]["access_log"] is False
    client = TestClient(served[0][0])
    evaluation_token = json.loads((state / "client.json").read_text())["token"]
    feedback_token = json.loads(reviewer_path.read_text())["token"]
    first_feedback = None
    for index in range(6):
        case = json.loads(json.dumps(body))
        case.update(caseId=f"case-{index}", sourceGroupId=f"source-{index}", idempotencyKey=f"test-key-{index}")
        case["input"]["text"] = f"Thank you for contacting support about order {index}."
        response = client.post("/v1/evaluate", json=case, headers={"Authorization": "Bearer " + evaluation_token})
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["outcome"] == "uncertain"
        feedback = {"workspaceId": case["workspaceId"], "evaluationId": result["inputCommitment"],
                    "inputCommitment": result["inputCommitment"], "templateCommitment": result["templateCommitment"],
                    "annotatorId": "reviewer-test", "labels": {
                        q["id"]: q["labels"][0]["id"] for q in case["template"]["questions"]},
                    "exposedToAi": False, "independentHuman": True}
        if first_feedback is None:
            first_feedback = feedback
            assert client.post("/v1/feedback", json=feedback,
                               headers={"Authorization": "Bearer " + evaluation_token}).status_code == 403
        reviewed = client.post("/v1/feedback", json=feedback, headers={"Authorization": "Bearer " + feedback_token})
        assert reviewed.status_code == 200, reviewed.text
        assert reviewed.json()["quarantineReasons"] == []
    created = invoke(capsys, state, "snapshot", "--template", template_id,
                     "--version", body["template"]["version"])
    assert created["groups"] == 6

    import rateloop_evaluator.training
    calls = []
    def train_double(store, snapshot_id, workspace_id, model_dir, output_dir, *, bundle_id, options):
        snapshot = store.load_snapshot(snapshot_id, workspace_id)
        calls.append((snapshot, bundle_id, options))
        return {"modelDir": str(output_dir), "training": {"optimizerSteps": 1}}
    monkeypatch.setattr(rateloop_evaluator.training, "train_snapshot", train_double)
    command = ("train", "--snapshot-id", created["snapshotId"], "--model-dir", model_dir,
               "--output", tmp_path / "trained", "--bundle-id", "adapted-test", "--method", "full", "--max-steps", "1")
    trained = invoke(capsys, state, *command)
    assert trained["optimizerSteps"] == 1 and trained["qualityClaim"] is False
    assert calls[0][1] == "adapted-test" and calls[0][2].method == "full"
    assert {r["group_id"] for r in calls[0][0]["train"]}.isdisjoint(
        {r["group_id"] for r in calls[0][0]["calibration"] + calls[0][0]["test"]})
    revoked = invoke(capsys, state, "revoke", "--grant-id", private["id"])
    assert created["snapshotId"] in revoked["invalidated_snapshots"]
    invoke(capsys, state, *command, expected=1)


def test_cli_adapted_registration_cannot_omit_lineage(initialized, tmp_path, capsys):
    request_path, body = request_file(tmp_path)
    model_dir = model_files(tmp_path, training={"bundleId": body["modelBundleId"],
        "workspaceId": body["workspaceId"], "snapshotId": "unregistered-snapshot"})
    invoke(capsys, initialized, "register", "--model-dir", model_dir, "--request", request_path, expected=1)


def test_cli_export_rejects_inactive_bundle(initialized, tmp_path, capsys):
    model_dir = model_files(tmp_path)
    request_a, body_a = request_file(tmp_path, "model-a")
    request_b, _ = request_file(tmp_path, "model-b")
    invoke(capsys, initialized, "register", "--model-dir", model_dir, "--request", request_a)
    invoke(capsys, initialized, "register", "--model-dir", model_dir, "--request", request_b)
    output = tmp_path / "registration.json"
    invoke(capsys, initialized, "export-registration", "--bundle-id", "model-a", "--request", request_a,
           "--output", output, expected=1)
    assert not output.exists()
    invoke(capsys, initialized, "export-registration", "--bundle-id", "model-b", "--request", request_b,
           "--output", output)
    registration = json.loads(output.read_text())
    assert registration["modelBundleId"] == "model-b"
    fixture = Path(__file__).parent / "fixtures" / "score-capability.json"
    assert registration["scoreCapability"] == json.loads(fixture.read_text())
    assert all(criterion["calibrationId"] is None for criterion in registration["criteria"])
    assert body_a["input"]["text"] not in output.read_text()
