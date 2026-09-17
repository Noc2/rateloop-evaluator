import hashlib
import json
import time
from pathlib import Path

from fastapi.testclient import TestClient
import pytest
from rateloop_evaluator.learning import LearningStore, provision_key
from rateloop_evaluator.protocol import EvaluationRequest
from rateloop_evaluator.service import Principal, create_app
from rateloop_evaluator.storage import RuntimeStore

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "test-token-with-enough-entropy-for-fixtures"


class TestBackend:
    __test__ = False
    calls = 0
    length = 100
    def count_tokens(self,text,questions): return self.length
    def predict(self,text,questions):
        self.calls += 1
        return {q["id"]:{label["id"]: 1/len(q["labels"]) for label in q["labels"]} for q in questions}


@pytest.fixture
def setup(tmp_path):
    key = tmp_path / "key"; provision_key(key)
    store = LearningStore(tmp_path / "learning", key)
    runtime = RuntimeStore(tmp_path / "runtime.sqlite", key)
    req = EvaluationRequest.model_validate_json((ROOT / "examples/reply-request.json").read_text())
    grant = store.add_grant(workspace_id=req.workspaceId,rights=["ai_use"],expires_at=time.time()+3600,evidence="synthetic test")
    backend = TestBackend()
    bundle = {"id":req.modelBundleId,"languages":["en","de"],"template_commitments":[req.template_commitment()],"max_tokens":512,"calibrations":[]}
    client = TestClient(create_app(backend=backend,bundle=bundle,learning=store,runtime=runtime,
        tokens={hashlib.sha256(TOKEN.encode()).hexdigest():Principal(req.workspaceId,frozenset({"evaluate","feedback"}),"human-test")}))
    client.headers["Authorization"] = "Bearer "+TOKEN
    return client,req.model_dump(),store,runtime,backend,grant


def test_authenticated_shadow_and_idempotency(setup):
    client,body,store,runtime,backend,_ = setup
    response = client.post("/v1/evaluate",json=body)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["outcome"] == "uncertain" and result["abstainReason"] == "uncalibrated"
    assert client.post("/v1/evaluate",json=body).json() == result and backend.calls == 1
    body["input"]["text"] = "Changed artifact"
    assert client.post("/v1/evaluate",json=body).status_code == 409
    with store.transaction() as state:
        assert all(row["input"] is None for row in state["evaluations"].values())
    assert b"synthetic-reply" not in runtime.path.read_bytes()


def test_tenant_auth_origin_and_revocation(setup):
    client,body,store,_,backend,grant = setup
    assert client.post("/v1/evaluate",json=body,headers={"Authorization":"Bearer wrong"}).status_code == 401
    assert client.post("/v1/evaluate",json=body,headers={"Origin":"https://untrusted.example"}).status_code == 403
    foreign = dict(body,workspaceId="another-workspace")
    assert client.post("/v1/evaluate",json=foreign).status_code == 403
    assert client.post("/v1/evaluate",json=body).status_code == 200
    store.revoke_grant(grant["id"],body["workspaceId"])
    assert client.post("/v1/evaluate",json=body).status_code == 403
    assert backend.calls == 1


def test_lengths_validation_and_private_feedback(setup):
    client,body,_,_,backend,_ = setup
    backend.length = 513
    result = client.post("/v1/evaluate",json=body).json()
    assert result["abstainReason"] == "input_too_long" and backend.calls == 0
    body["input"]["secret"] = "NEVER-ECHO-THIS"
    invalid = client.post("/v1/evaluate",json=body)
    assert invalid.status_code == 422 and "NEVER-ECHO-THIS" not in invalid.text
    oversized = client.post("/v1/evaluate",content=b"x"*350001)
    assert oversized.status_code == 413
    feedback = {"workspaceId":body["workspaceId"],"evaluationId":result["inputCommitment"],"inputCommitment":result["inputCommitment"],"templateCommitment":result["templateCommitment"],"annotatorId":"human-test", "labels":{"tone":"suitable"},"exposedToAi":False,"independentHuman":True}
    assert client.post("/v1/feedback",json=feedback).status_code == 403


def test_private_training_retention_and_feedback(setup):
    client,body,store,_,_,_ = setup
    store.add_grant(workspace_id=body["workspaceId"],rights=["private_training"],expires_at=time.time()+3600,evidence="synthetic independent labels")
    result = client.post("/v1/evaluate",json=body).json()
    feedback = {"workspaceId":body["workspaceId"],"evaluationId":result["inputCommitment"],"inputCommitment":result["inputCommitment"],"templateCommitment":result["templateCommitment"],"annotatorId":"human-test", "labels":{"tone":"suitable"},"exposedToAi":False,"independentHuman":True}
    response = client.post("/v1/feedback",json=feedback)
    assert response.status_code == 200, response.text
    assert response.json()["quarantineReasons"] == []
    assert "duplicate_reviewer" in client.post("/v1/feedback",json=feedback).json()["quarantineReasons"]


def test_outbox_is_encrypted_and_retries_survive_reopen(setup,tmp_path):
    _,_,_,store,_,_ = setup
    payload = {"onlyApprovedMetadata":"opaque-reference"}
    store.enqueue("receipt-1",payload); store.enqueue("receipt-1",payload)
    assert store.pending() == [("receipt-1",payload)]
    assert b"opaque-reference" not in store.path.read_bytes()
    with pytest.raises(ValueError): store.enqueue("receipt-1",{"different":True})
    store.retry("receipt-1"); assert store.pending() == []
    store.delivered("receipt-1"); assert store.pending() == []


def test_reviewer_cannot_impersonate_another_human(setup):
    client,body,store,_,_,_ = setup
    store.add_grant(workspace_id=body["workspaceId"],rights=["private_training"],expires_at=time.time()+3600,evidence="synthetic independent labels")
    result = client.post("/v1/evaluate",json=body).json()
    feedback = {"workspaceId":body["workspaceId"],"evaluationId":result["inputCommitment"],"inputCommitment":result["inputCommitment"],"templateCommitment":result["templateCommitment"],"annotatorId":"another-person", "labels":{"tone":"suitable"},"exposedToAi":False,"independentHuman":True}
    assert client.post("/v1/feedback",json=feedback).status_code == 403


def test_changed_deployment_cannot_replay_cached_result(setup):
    _,body,store,runtime,backend,_ = setup
    request = EvaluationRequest.model_validate(body)
    deployment = {"mode":"shadow","revision":1}
    bundle = {"id":request.modelBundleId,"languages":["en"],"template_commitments":[request.template_commitment()],"max_tokens":512,"calibrations":[]}
    client = TestClient(create_app(backend=backend,bundle=bundle,learning=store,runtime=runtime,
        tokens={hashlib.sha256(TOKEN.encode()).hexdigest():Principal(request.workspaceId,frozenset({"evaluate"}))},
        validate_bundle=lambda _:dict(deployment)))
    client.headers["Authorization"] = "Bearer "+TOKEN
    assert client.post("/v1/evaluate",json=body).status_code == 200
    deployment["revision"] = 2
    assert client.post("/v1/evaluate",json=body).status_code == 409
    assert backend.calls == 1


def test_case_erasure_clears_runtime_and_prevents_inflight_reinsertion(setup):
    client,body,store,runtime,_,_ = setup
    result = client.post("/v1/evaluate",json=body).json()
    runtime.enqueue("delete-receipt",{"result":result})
    runtime.acknowledge("acked-receipt",{"workspaceId":body["workspaceId"],"caseId":body["caseId"],"receiptId":"server-receipt"})
    assert runtime.delete_case(body["workspaceId"],body["caseId"]) == {"results":1,"outbox":1,"acknowledgments":1}
    assert runtime.pending() == []
    with pytest.raises(PermissionError): runtime.enqueue("later-receipt",{"result":result})
    with pytest.raises(PermissionError): runtime.put(body["workspaceId"],body["idempotencyKey"],result["inputCommitment"],{"result":result})
