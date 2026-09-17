from copy import deepcopy
import hashlib
import json
import plistlib
from pathlib import Path
import time

import httpx
import pytest

from test_connector import setup, iso, review_context
from test_durable_consent import durable
from rateloop_evaluator.connector import RateLoopConnector, ConnectorUnavailable
from rateloop_evaluator.protocol import EvaluationRequest
from rateloop_evaluator.service import Principal, create_app
from rateloop_evaluator.templates import overall_approval
from rateloop_evaluator.worker import OutboundWorker, install_launchd, single_worker


@pytest.fixture
def website(setup):
    original,req,learning,runtime,remote,behavior,calls,kwargs=setup
    req=req.model_copy(deep=True); req.template=overall_approval(); req.input.evidence=""
    durable(remote,req,time.time())
    ai=deepcopy(remote["consents"][0]); ai.update(consentId="ai-permission",purpose="ai_use")
    remote["consents"].append(ai)
    job={"jobId":"job-1","modelBundleId":req.modelBundleId,"inputCommitment":req.input_commitment(),
        "templateCommitment":req.template_commitment(),"leaseToken":"secret-fencing-token-"+"a"*32,"leaseExpiresAt":iso(time.time()+120)}
    audit={"auditId":"audit-job-1","selected":True,"kind":"mandatory","aiExposed":False,"selectionProbabilityBps":10000,
        "blindingAssurance":"server_enforced","selectedAt":iso(time.time()),"sourceContentHash":"sha256:"+hashlib.sha256(req.input.context.encode()).hexdigest(),
        "suggestedContentHash":"sha256:"+hashlib.sha256(req.input.text.encode()).hexdigest(),"frozenQuestionHash":"sha256:"+"d"*64}
    body={"request":req.model_dump(),"reviewContext":review_context(),"audit":audit,"agentId":"agent-1","agentVersionId":"version-1",
          "createdAt":iso(time.time()),"retainForTraining":True}
    behavior.update(claimed=False,completed=False,heartbeat_status=200,complete_status=200,content=body)
    old_transport=kwargs["transport"]
    def handle(request):
        path=request.url.path
        if path.endswith("/workers/heartbeat"):
            calls.append(request)
            payload=json.loads(request.content)
            return httpx.Response(200,json={"workerId":payload["workerId"],"state":payload["state"],"lastSeenAt":iso(time.time())})
        if "/jobs/" not in path: return old_transport.handle_request(request)
        calls.append(request)
        if path.endswith("/claim"):
            if behavior["claimed"]: return httpx.Response(200,json={"job":None})
            behavior["claimed"]=True
            return httpx.Response(200,json={"job":job})
        if path.endswith("/content"):
            assert request.headers["x-evaluator-lease"]==job["leaseToken"]
            assert request.headers["x-evaluator-worker"]=="mac-1"
            return httpx.Response(200,json=behavior["content"])
        payload=json.loads(request.content)
        assert payload["leaseToken"]==job["leaseToken"] and payload["workerId"]=="mac-1"
        if path.endswith("/heartbeat"):
            return httpx.Response(behavior["heartbeat_status"],json={"leaseExpiresAt":iso(time.time()+120)})
        if path.endswith("/complete"):
            assert payload["receiptId"]=="aev_"+"1"*40
            behavior["completed"]=True
            return httpx.Response(behavior["complete_status"],json={"state":"awaiting_human_review"})
        if path.endswith("/fail"):
            behavior["failed"]=payload
            return httpx.Response(200,json={"state":"failed"})
        raise AssertionError(path)
    connector=RateLoopConnector(**{**kwargs,"transport":httpx.MockTransport(handle)},metadata_upload_enabled=True)
    class Backend:
        calls=0
        def count_tokens(self,*_): return 40
        def predict(self,*_):
            self.calls+=1
            return {"overall_approval":{"approved":.8,"rejected":.2}}
    backend=Backend()
    identity=Principal(req.workspaceId,frozenset({"evaluate"}))
    def allow_retention(request):
        with learning.transaction() as db:
            return connector._state(db).get("collections",{}).get(request.input_commitment(),{}).get("trainingAllowed") is True
    app=create_app(backend=backend,bundle={"id":req.modelBundleId,"languages":["en"],"template_commitments":[req.template_commitment()]},
        learning=learning,runtime=runtime,tokens={"0"*64:identity},allow_training_retention=allow_retention)
    worker=OutboundWorker(connector,worker_id="mac-1",model_bundle_ids=[req.modelBundleId],evaluate=lambda r:app.state.evaluate(r,identity))
    return worker,req,backend,remote,behavior,calls


def test_website_job_uses_shared_inference_core_and_encrypts_progress(website):
    worker,req,backend,_,behavior,calls=website
    behavior["blind_receipt"]=True
    result=worker.run_once()
    assert result=={"state":"completed","jobId":"job-1","modelBundleId":req.modelBundleId,"humanReviewRequired":True}
    assert backend.calls==1 and behavior["completed"]
    assert worker._saved() is None
    assert worker.run_once()=={"state":"idle"}
    receipt=next(json.loads(r.content) for r in calls if r.url.path.endswith("/receipts"))
    receipt_request=next(r for r in calls if r.url.path.endswith("/receipts"))
    assert receipt_request.headers["x-evaluator-job"]=="job-1" and receipt_request.headers["x-evaluator-worker"]=="mac-1"
    assert receipt_request.headers["x-evaluator-lease"].startswith("secret-fencing-token")
    assert receipt["result"]["criteria"][0]["label"]=="approved"
    assert req.input.text not in json.dumps(receipt)
    assert req.input.text.encode() not in worker.connector.learning._path.read_bytes()
    assert req.input.text.encode() not in worker.connector.runtime.path.read_bytes()
    with worker.connector.learning.transaction() as db:
        assert db["evaluations"][req.input_commitment()]["input"]["text"]==req.input.text


def test_inference_only_permission_does_not_retain_training_inputs(website):
    worker,req,_,remote,_,_=website
    remote["consents"]=[c for c in remote["consents"] if c["purpose"]=="ai_use"]
    assert worker.run_once()["state"]=="completed"
    with worker.connector.learning.transaction() as db:
        assert db["evaluations"][req.input_commitment()]["input"] is None


def test_later_opt_in_cannot_retain_an_earlier_queued_case(website):
    worker,req,_,_,behavior,_=website
    behavior["content"]["createdAt"]=iso(time.time()-100)
    assert worker.run_once()["state"]=="completed"
    with worker.connector.learning.transaction() as db:
        assert db["evaluations"][req.input_commitment()]["input"] is None


def test_explicit_case_tombstone_purges_retained_case_and_acknowledgment(website):
    worker,req,_,remote,_,_=website
    worker.run_once()
    remote.update(deletedCases=[{"caseId":req.caseId,"deletedAt":iso(time.time())}],deletionWatermark=1)
    worker.connector.sync_grants()
    with worker.connector.learning.transaction() as db:
        assert req.input_commitment() not in db["evaluations"]
        assert not worker.connector._state(db)["results"]
    with worker.connector.runtime.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM acknowledgments").fetchone()[0]==0
    remote["deletionWatermark"]=0
    with pytest.raises(PermissionError,match="Deletion watermark"):
        worker.connector.sync_grants()


def test_workspace_notice_erases_only_explicit_case_ids(website):
    worker,req,_,remote,_,_=website
    worker.run_once()
    unrelated=req.model_copy(deep=True);unrelated.caseId="locally-originated-case"
    worker.connector.learning.record_evaluation(evaluation_id=unrelated.input_commitment(),workspace_id=req.workspaceId,
        case_id=unrelated.caseId,input_commitment=unrelated.input_commitment(),template_commitment=req.template_commitment(),
        template=req.template.model_dump(),input_payload=req.input.model_dump(),model_bundle_id=req.modelBundleId)
    remote.update(workspaceDeletion={"deletedAt":iso(time.time())},deletedCases=[{"caseId":req.caseId,"deletedAt":iso(time.time())}],deletionWatermark=1)
    result=worker.connector.sync_grants()
    assert result["workspaceDeleted"] and result["mode"]=="off"
    with worker.connector.learning.transaction() as db:
        assert req.input_commitment() not in db["evaluations"]
        assert db["evaluations"][unrelated.input_commitment()]["input"] is not None
    del remote["workspaceDeletion"]
    with pytest.raises(PermissionError,match="deleted workspace"):
        worker.connector.sync_grants()


def test_restart_recovers_fenced_job_and_durable_receipt_without_rescoring(website):
    worker,req,backend,_,behavior,calls=website
    behavior["receipt_status"]=503
    with pytest.raises(ConnectorUnavailable): worker.run_once()
    assert worker._saved()["jobId"]=="job-1"
    encrypted=worker.connector.learning._path.read_bytes()
    assert b"secret-fencing-token" not in encrypted
    assert backend.calls==1
    behavior["receipt_status"]=200
    with worker.connector.runtime.connect() as db: db.execute("UPDATE outbox SET next_attempt=0")
    restarted=OutboundWorker(worker.connector,worker_id="mac-1",model_bundle_ids=[req.modelBundleId],evaluate=worker.evaluate)
    assert restarted.run_once()["state"]=="completed"
    assert backend.calls==1
    assert len([r for r in calls if r.url.path.endswith("/claim")])==1


def test_lost_completion_ack_does_not_resubmit_a_completed_case(website):
    worker,_,backend,_,behavior,_=website
    behavior["complete_status"]=503
    with pytest.raises(ConnectorUnavailable): worker.run_once()
    assert behavior["completed"] and worker._saved()
    behavior["heartbeat_status"]=409
    assert worker.run_once()=={"state":"idle"}
    assert worker._saved() is None and backend.calls==1


def test_retry_after_receipt_ack_does_not_requeue_delivered_metadata(website):
    worker,_,backend,_,behavior,calls=website
    behavior["complete_status"]=503
    with pytest.raises(ConnectorUnavailable): worker.run_once()
    behavior["complete_status"]=200
    assert worker.run_once()["state"]=="completed"
    with worker.connector.runtime.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]==0
    assert backend.calls==1
    assert len([r for r in calls if r.url.path.endswith("/receipts")])==1


@pytest.mark.parametrize("change",["source","question","agent"])
def test_changed_content_or_review_identity_rejected_before_inference(website,change):
    worker,_,backend,_,behavior,_=website
    if change=="source": behavior["content"]["request"]["input"]["context"]+=" tampered"
    if change=="question": behavior["content"]["audit"]["kind"]="failure_only"
    if change=="agent": behavior["content"]["agentVersionId"]="different-agent"
    assert worker.run_once()["state"]=="failed"
    assert backend.calls==0 and behavior["failed"]["retryable"] is False


def test_paused_workspace_never_claims_content(website):
    worker,_,backend,remote,_,calls=website
    remote["settings"]["mode"]="paused"
    assert worker.run_once()=={"state":"paused"}
    assert backend.calls==0 and not any("/jobs/" in r.url.path for r in calls)


def test_launchd_install_has_only_private_config_path_not_credentials(tmp_path,monkeypatch):
    monkeypatch.setattr("sys.platform","darwin")
    config=tmp_path/"connector.json"; config.write_text('{"apiKey":"private-api-key"}'); config.chmod(0o600)
    output=tmp_path/"worker.plist"
    result=install_launchd(state_dir=tmp_path,config_path=str(config),worker_id="mac-1",bundle_ids=["model-1"],
        device="mps",poll_seconds=5,output=str(output))
    contents=output.read_bytes(); value=plistlib.loads(contents)
    assert b"private-api-key" not in contents and result["loaded"] is False
    assert value["ProgramArguments"][-2:]==["--bundle-id","model-1"]
    assert output.stat().st_mode&0o777==0o600
    with pytest.raises(FileExistsError):
        install_launchd(state_dir=tmp_path,config_path=str(config),worker_id="mac-1",bundle_ids=["model-1"],
            device="mps",poll_seconds=5,output=str(output))


def test_single_worker_prevents_concurrent_duplicate_identity(tmp_path):
    with single_worker(tmp_path,"worker-1"):
        with pytest.raises(RuntimeError,match="already running"):
            with single_worker(tmp_path,"worker-1"): pass


def test_frozen_template_examples_match_python_contract():
    for language in ("en","de"):
        path=Path(__file__).parents[1]/f"examples/approval-request-{language}.json"
        request=EvaluationRequest.model_validate_json(path.read_text())
        assert request.template==overall_approval(language)
