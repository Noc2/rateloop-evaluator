from copy import deepcopy
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
import json
import plistlib
from pathlib import Path
import time

import httpx
import pytest

from test_connector import setup, iso, review_context
from test_durable_consent import durable
from rateloop_evaluator.connector import RateLoopConnector, ConnectorUnavailable
from rateloop_evaluator.protocol import EvaluationRequest, commitment, utc_now
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
    behavior.update(claimed=False,completed=False,heartbeat_status=200,complete_status=200,content=body,job=job)
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
            return httpx.Response(behavior["complete_status"],json={"state":"completed" if job.get("reviewMode")=="ai" else "awaiting_human_review"})
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


def choose_ai_only(behavior):
    behavior["job"]["reviewMode"]="ai"
    behavior["content"].update(reviewMode="ai",audit=None,retainForTraining=False)


def test_explicit_ai_only_returns_advisory_prediction_without_human_or_training_reference(website):
    worker,req,backend,_,behavior,calls=website
    choose_ai_only(behavior)
    assert worker.run_once()=={"state":"completed","jobId":"job-1","modelBundleId":req.modelBundleId,"humanReviewRequired":False}
    assert backend.calls==1 and behavior["completed"]
    assert not any(r.url.path.endswith("/audits") for r in calls)
    receipt=next(json.loads(r.content) for r in calls if r.url.path.endswith("/receipts"))
    assert receipt["result"]["outcome"]=="uncertain"
    assert receipt["result"]["abstainReason"]=="uncalibrated"
    assert receipt["result"]["criteria"][0]["label"]=="approved"
    with worker.connector.learning.transaction() as db:
        state=worker.connector._state(db)
        assert req.input_commitment() not in state["audits"]
        assert state["collections"][req.input_commitment()]["reviewMode"]=="ai"
        assert state["collections"][req.input_commitment()]["trainingAllowed"] is False
        assert db["evaluations"][req.input_commitment()]["input"] is None
        assert not db["feedback"]
    with pytest.raises(PermissionError,match="AI-only cases"):
        worker.connector.learning.add_feedback(workspace_id=req.workspaceId,evaluation_id=req.input_commitment(),
            input_commitment=req.input_commitment(),template_commitment=req.template_commitment(),
            annotator_id="human-1",labels={"overall_approval":"approved"},exposed_to_ai=False,independent_human=True)


@pytest.mark.parametrize("mode",["missing","ai_and_human"])
def test_legacy_and_explicit_both_require_blinded_human_review(website,mode):
    worker,req,_,_,behavior,_=website
    if mode!="missing":
        behavior["job"]["reviewMode"]=mode
        behavior["content"]["reviewMode"]=mode
    assert worker.run_once()["humanReviewRequired"] is True
    with worker.connector.learning.transaction() as db:
        assert worker.connector._state(db)["audits"][req.input_commitment()]["response"]["selected"] is True


@pytest.mark.parametrize("claim_mode,content_mode",[
    ("ai","ai_and_human"),("ai_and_human","ai"),("missing","ai"),("ai","missing"),
    ("ai",None),("ai","human"),("ai","auto_approve"),
])
def test_content_cannot_change_claimed_review_selection(website,claim_mode,content_mode):
    worker,_,backend,_,behavior,_=website
    if claim_mode!="missing": behavior["job"]["reviewMode"]=claim_mode
    if content_mode!="missing": behavior["content"]["reviewMode"]=content_mode
    assert worker.run_once()["state"]=="failed"
    assert backend.calls==0


@pytest.mark.parametrize("mode",[None,"human","auto_approve",True])
def test_claim_rejects_unknown_or_human_only_modes(website,mode):
    worker,_,backend,_,behavior,calls=website
    behavior["job"]["reviewMode"]=mode
    with pytest.raises(ValueError,match="review mode"): worker.run_once()
    assert backend.calls==0 and not any(r.url.path.endswith("/content") for r in calls)


@pytest.mark.parametrize("change",["audit","missing_audit","retention","missing_retention"])
def test_ai_only_rejects_human_or_training_provenance(website,change):
    worker,_,backend,_,behavior,_=website
    audit=deepcopy(behavior["content"]["audit"])
    choose_ai_only(behavior)
    if change=="audit": behavior["content"]["audit"]=audit
    if change=="missing_audit": del behavior["content"]["audit"]
    if change=="retention": behavior["content"]["retainForTraining"]=True
    if change=="missing_retention": del behavior["content"]["retainForTraining"]
    assert worker.run_once()["state"]=="failed" and backend.calls==0


@pytest.mark.parametrize("change",["absent","null","selected","kind","exposed","probability","blinding"])
def test_both_modes_keep_every_human_blinding_requirement(website,change):
    worker,_,backend,_,behavior,_=website
    behavior["job"]["reviewMode"]="ai_and_human"
    behavior["content"]["reviewMode"]="ai_and_human"
    if change=="absent": del behavior["content"]["audit"]
    elif change=="null": behavior["content"]["audit"]=None
    else:
        key,value={"selected":("selected",False),"kind":("kind","random"),"exposed":("aiExposed",True),
            "probability":("selectionProbabilityBps",5000),"blinding":("blindingAssurance","connector_attested")}[change]
        behavior["content"]["audit"][key]=value
    assert worker.run_once()["state"]=="failed" and backend.calls==0


@pytest.mark.parametrize("first_mode",["ai","ai_and_human"])
def test_retry_cannot_reclassify_a_scored_case_as_a_different_review_mode(website,first_mode):
    worker,_,backend,_,behavior,_=website
    original_audit=deepcopy(behavior["content"]["audit"])
    if first_mode=="ai": choose_ai_only(behavior)
    behavior["complete_status"]=503
    with pytest.raises(ConnectorUnavailable): worker.run_once()
    saved=worker._saved()
    saved["reviewMode"]="ai_and_human" if first_mode=="ai" else "ai"
    worker._save(saved)
    behavior["content"]["reviewMode"]=saved["reviewMode"]
    behavior["content"]["audit"]=original_audit if saved["reviewMode"]=="ai_and_human" else None
    behavior["content"]["retainForTraining"]=saved["reviewMode"]=="ai_and_human"
    assert worker.run_once()["state"]=="failed"
    assert backend.calls==1


def test_ai_only_receipt_retry_keeps_choice_and_does_not_score_twice(website):
    worker,req,backend,_,behavior,_=website
    choose_ai_only(behavior); behavior["receipt_status"]=503
    with pytest.raises(ConnectorUnavailable): worker.run_once()
    assert worker._saved()["reviewMode"]=="ai"
    behavior["receipt_status"]=200
    with worker.connector.runtime.connect() as db: db.execute("UPDATE outbox SET next_attempt=0")
    restarted=OutboundWorker(worker.connector,worker_id="mac-1",model_bundle_ids=[req.modelBundleId],evaluate=worker.evaluate)
    assert restarted.run_once()["humanReviewRequired"] is False
    assert backend.calls==1


def test_ai_only_label_export_is_rejected_even_with_forged_human_provenance(website):
    worker,req,_,remote,behavior,_=website
    choose_ai_only(behavior); worker.run_once()
    body={"schemaVersion":"rateloop.evaluator-labeled-data.v2","workspaceId":req.workspaceId,
        "consent":deepcopy(remote["consents"][0]),"revocationWatermark":remote["revocationWatermark"],
        "contentMode":"commitments_only","window":{"from":iso(time.time()-3600),"to":utc_now()},
        "items":[{"inputCommitment":req.input_commitment(),"audit":{"independent":True}}],"truncated":False}
    behavior["labels"]={**body,"exportDigest":commitment(body,"rateloop.product-evaluator.v2")}
    imported=worker.connector.fetch_and_import_labels("permission-1",question_id="overall_approval",
        template_commitment=req.template_commitment(),outcome_labels={"positive":"approved","negative":"rejected"})
    assert imported["imported"]==0
    assert imported["rejected"][0]["reason"]=="AI-only cases have no independent human training reference"
    with worker.connector.learning.transaction() as db: assert not db["feedback"]


def test_acceptance_operator_inspects_ai_only_privacy_without_exporting_inputs(website):
    worker,req,_,_,behavior,_=website
    choose_ai_only(behavior); worker.run_once()
    spec=spec_from_file_location("alpha_operator_review_test",Path(__file__).parents[1]/"scripts/alpha_e2e_operator.py")
    operator=module_from_spec(spec);spec.loader.exec_module(operator)
    assert operator.verify_ai_only_case(worker.connector,req.caseId)=={
        "caseId":req.caseId,"aiOnly":True,"retainedTrainingInputs":0,"humanAudits":0,"humanLabels":0}
    with pytest.raises(ValueError,match="one exact local"): operator.verify_ai_only_case(worker.connector,"missing")
    with worker.connector.learning.transaction() as db:
        db["evaluations"][req.input_commitment()]["input"]={"text":"private wrong retention"}
    leaked=operator.verify_ai_only_case(worker.connector,req.caseId)
    assert leaked["retainedTrainingInputs"]==1 and "private wrong retention" not in json.dumps(leaked)
    with worker.connector.learning.transaction() as db:
        worker.connector._state(db)["collections"][req.input_commitment()]["reviewMode"]="ai_and_human"
    with pytest.raises(ValueError,match="not AI-only"): operator.verify_ai_only_case(worker.connector,req.caseId)


def test_later_opt_in_cannot_retain_an_earlier_queued_case(website):
    worker,req,_,_,behavior,_=website
    behavior["content"]["createdAt"]=iso(time.time()-100)
    assert worker.run_once()["state"]=="completed"
    with worker.connector.learning.transaction() as db:
        assert db["evaluations"][req.input_commitment()]["input"] is None


@pytest.mark.parametrize("mode",["ai","ai_and_human"])
def test_explicit_case_tombstone_purges_retained_case_and_acknowledgment(website,mode):
    worker,req,_,remote,behavior,_=website
    if mode=="ai": choose_ai_only(behavior)
    worker.run_once()
    remote.update(deletedCases=[{"caseId":req.caseId,"deletedAt":iso(time.time())}],deletionWatermark=1)
    worker.connector.sync_grants()
    with worker.connector.learning.transaction() as db:
        assert req.input_commitment() not in db["evaluations"]
        assert not worker.connector._state(db)["results"]
        assert not worker.connector._state(db)["collections"]
    with worker.connector.runtime.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM acknowledgments").fetchone()[0]==0
    remote["deletionWatermark"]=0
    with pytest.raises(PermissionError,match="Deletion watermark"):
        worker.connector.sync_grants()


@pytest.mark.parametrize("mode",["ai","ai_and_human"])
def test_deleted_case_cannot_recreate_collection_metadata_or_score(website,mode):
    worker,req,backend,remote,behavior,_=website
    if mode=="ai": choose_ai_only(behavior)
    remote.update(deletedCases=[{"caseId":req.caseId,"deletedAt":iso(time.time())}],deletionWatermark=1)
    assert worker.run_once()["state"]=="failed" and backend.calls==0
    with worker.connector.learning.transaction() as db:
        assert not worker.connector._state(db).get("collections")
        assert not worker.connector._state(db)["audits"]


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
