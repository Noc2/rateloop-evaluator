"""Actual connector CLI routing with bounded HTTP transports and private config."""
from argparse import Namespace
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import httpx
import pytest

from rateloop_evaluator import cli
from rateloop_evaluator.protocol import EvaluationRequest, commitment, make_result, utc_now

ROOT = Path(__file__).resolve().parents[1]


def iso(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def invoke(capsys, state, *arguments, expected=0):
    result = cli.main(["--state-dir", str(state), *map(str, arguments)])
    output = capsys.readouterr()
    assert result == expected, output.err
    return output.err if expected else json.loads(output.out)


@pytest.fixture
def connected(tmp_path, capsys, monkeypatch):
    request = EvaluationRequest.model_validate_json((ROOT / "examples/reply-request.json").read_text())
    state = tmp_path / "state"
    invoke(capsys, state, "init", "--workspace", request.workspaceId)
    invoke(capsys, state, "grant", "--right", "ai_use", "--template", request.template.id, "--evidence", "Synthetic CLI test")
    _, _, store, _ = cli.state(Namespace(state_dir=str(state)))
    config_path = tmp_path / "connector.json"
    config = {"baseUrl": "https://rateloop.example", "apiKey": "private-test-api-key", "apiKeyId": "api-1",
              "agentId": "agent-1", "agentVersionId": "agent-version-1", "metadataUploadEnabled": True}
    cli.write_private(config_path, config)
    request_path = tmp_path / "request.json"
    request_path.write_text(request.model_dump_json())
    context_path = tmp_path / "context.json"
    context_path.write_text(json.dumps({"policyId": "policy-1", "policyVersion": 1, "workflowKey": "reply",
        "riskTier": "low", "audiencePolicyHash": "sha256:" + "a" * 64, "metadataComplete": True,
        "execution": {"externalExecutionId": "execution-1", "status": "completed", "primarySpanId": "span-1",
                      "generationSpans": [{"spanId": "span-1", "role": "primary", "provider": "local", "requestedModel": "fixture"}]}}))
    grant = {"grantId": "grant-remote", "workspaceId": request.workspaceId, "apiKeyId": "api-1", "purpose": "private_learning",
             "modelBundleId": request.modelBundleId, "templateCommitment": request.template_commitment(),
             "fields": ["input", "context", "evidence", "human_labels"], "publicWeightsAllowed": False,
             "issuedAt": iso(time.time() - 10), "expiresAt": iso(time.time() + 3600), "revokedAt": None, "revision": 1}
    behavior = {"mode": "shadow", "unavailable": False}
    calls = []

    def remote(http_request):
        calls.append(("remote", http_request.url.path))
        assert http_request.headers["authorization"] == "Bearer private-test-api-key"
        if behavior["unavailable"]:
            raise httpx.ConnectError("synthetic offline", request=http_request)
        if http_request.url.path.endswith("/grants"):
            return httpx.Response(200, json={"workspaceId": request.workspaceId, "recipientApiKeyId": "api-1",
                "revocationWatermark": 1, "settings": {"mode": behavior["mode"]}, "grants": [grant]})
        if http_request.url.path.endswith("/audits"):
            return httpx.Response(200, json={"auditId": "audit-1", "kind": "random", "selected": True,
                "aiExposed": False, "selectionProbabilityBps": 1000, "blindingAssurance": "connector_attested"})
        if http_request.url.path.endswith("/receipts"):
            receipt = json.loads(http_request.content)
            assert request.input.text not in http_request.content.decode()
            return httpx.Response(200, json={"schemaVersion": "rateloop.automated-eval-ingest-result.v2",
                "receiptId": "aev_" + "1" * 40, "receiptHash": commitment(receipt, "rateloop.product-evaluator.v2"),
                "outcome": receipt["result"]["outcome"], "policy": {"mayReduceHumanReview": False}, "replayed": False})
        if http_request.url.path.endswith("/labeled-data"):
            assert http_request.url.params["grantId"] == grant["grantId"]
            body = {"schemaVersion": "rateloop.evaluator-labeled-data.v2", "workspaceId": request.workspaceId,
                    "grant": grant, "revocationWatermark": 1, "contentMode": "commitments_only", "items": [], "truncated": False}
            return httpx.Response(200, json={**body, "exportDigest": commitment(body, "rateloop.product-evaluator.v2")})
        raise AssertionError(http_request.url)

    def local(http_request):
        calls.append(("local", http_request.url.path))
        expected = json.loads((state / "client.json").read_text())["token"]
        assert http_request.headers["authorization"] == "Bearer " + expected
        value = EvaluationRequest.model_validate_json(http_request.content)
        store.record_evaluation(evaluation_id=value.input_commitment(), workspace_id=value.workspaceId,
            case_id=value.caseId, input_commitment=value.input_commitment(), template_commitment=value.template_commitment(),
            template=value.template.model_dump(), input_payload=value.input.model_dump(),
            model_bundle_id=value.modelBundleId, group_id=value.sourceGroupId)
        result = make_result(workspaceId=value.workspaceId, caseId=value.caseId, modelBundleId=value.modelBundleId,
            inputCommitment=value.input_commitment(), templateCommitment=value.template_commitment(), outcome="uncertain",
            abstainReason="uncalibrated", criteria=[{"questionId": q.id, "label": q.labels[0].id,
                "rawScores": {label.id: 1 / len(q.labels) for label in q.labels}, "probabilities": None, "calibrationId": None}
                for q in value.template.questions], durationMs=10, observedAt=utc_now())
        return httpx.Response(200, json=result.model_dump())

    original_client = httpx.Client
    def client_factory(*args, **kwargs):
        assert kwargs.get("trust_env") is False and kwargs.get("follow_redirects") is False
        origin = kwargs["base_url"]
        assert origin in ("https://rateloop.example", "http://127.0.0.1:8765")
        kwargs["transport"] = httpx.MockTransport(remote if origin == "https://rateloop.example" else local)
        return original_client(*args, **kwargs)
    monkeypatch.setattr(httpx, "Client", client_factory)
    return state, config_path, config, request, request_path, context_path, behavior, calls


def evaluation_args(config_path, request_path, context_path):
    return ("connect-evaluate", "--config", config_path, "--request", request_path, "--review-context", context_path, "--frozen-question-hash", "sha256:"+"e"*64)


def test_cli_connector_config_requires_private_permissions_and_explicit_opt_in(connected, capsys):
    state, path, config, _, request_path, context_path, _, calls = connected
    path.chmod(0o644)
    invoke(capsys, state, "connect-sync", "--config", path, expected=1)
    assert not calls
    path.chmod(0o600)
    config.pop("metadataUploadEnabled")
    cli.write_private(path, config)
    invoke(capsys, state, "connect-sync", "--config", path, expected=1)
    assert not calls
    config["metadataUploadEnabled"] = False
    cli.write_private(path, config)
    invoke(capsys, state, "connect-flush", "--config", path, expected=1)
    assert not calls
    invoke(capsys, state, *evaluation_args(path, request_path, context_path), expected=1)
    assert not any(kind == "local" for kind, _ in calls)


@pytest.mark.parametrize("condition", ["paused", "off", "unavailable", "wrong-local-workspace"])
def test_cli_connector_refuses_inference_when_connection_or_identity_is_not_eligible(connected, capsys, tmp_path, condition):
    state, path, _, _, request_path, context_path, behavior, calls = connected
    arguments = evaluation_args(path, request_path, context_path)
    if condition in ("paused", "off"):
        behavior["mode"] = condition
    elif condition == "unavailable":
        behavior["unavailable"] = True
    else:
        credentials = tmp_path / "wrong-client.json"
        cli.write_private(credentials, {"workspaceId": "another-workspace", "token": "wrong-scope"})
        arguments += ("--client-credentials", credentials)
    invoke(capsys, state, *arguments, expected=1)
    assert not any(kind == "local" for kind, _ in calls)


def test_cli_connector_sync_evaluate_flush_import_and_release(connected, capsys):
    state, path, _, request, request_path, context_path, _, calls = connected
    synced = invoke(capsys, state, "connect-sync", "--config", path)
    assert synced["mirroredGrants"] == 1 and synced["mode"] == "shadow"
    evaluated = invoke(capsys, state, *evaluation_args(path, request_path, context_path))
    assert evaluated["awaitingIndependentHuman"] is True and evaluated["result"] is None
    assert sum(kind == "local" for kind, _ in calls) == 1
    flushed = invoke(capsys, state, "connect-flush", "--config", path)
    assert flushed == {"delivered": 1, "retrying": 0, "rejected": 0}
    imported = invoke(capsys, state, "connect-import-labels", "--config", path, "--grant-id", "grant-remote",
        "--question-id", "tone", "--template-commitment", request.template_commitment(),
        "--positive-label", "suitable", "--negative-label", "unsuitable")
    assert imported["imported"] == 0 and imported["rejected"] == []
    released = invoke(capsys, state, "connect-release", "--config", path, "--input-commitment", request.input_commitment())
    assert released["inputCommitment"] == request.input_commitment()
    assert released["outcome"] == "uncertain"
