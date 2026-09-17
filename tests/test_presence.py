"""Connected presence stays subordinate to execution locks and consent."""
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import threading
import time

import httpx
import pytest

from rateloop_evaluator import cli
from rateloop_evaluator.connector import ConnectorUnavailable
from rateloop_evaluator.execution import ExecutionBusy, model_execution
from rateloop_evaluator.presence import connected_training
from test_connector import setup, iso
from test_worker import website


@pytest.fixture
def presence(website):
    worker,request,backend,remote,behavior,calls=website
    original=worker.connector.client._transport
    state={"state":"ready","operationId":None,"lastSeenAt":iso(time.time()),"reports":[]}
    renewed=threading.Event()
    def handle(request):
        if not request.url.path.endswith("/workers/heartbeat"):
            return original.handle_request(request)
        body=json.loads(request.content);calls.append(request);state["reports"].append(body)
        if body["state"] == "ready" and state.get("readyUnavailable"):
            return httpx.Response(503,json={"code":"temporarily_unavailable"})
        current=state["operationId"]; operation=body.get("operationId")
        if current and operation and operation!=current:
            return httpx.Response(409,json={"code":"evaluator_worker_operation_conflict"})
        if not current or operation==current:
            state.update(state=body["state"],operationId=operation if body["state"]=="training" else None,lastSeenAt=iso(time.time()))
        if sum(r["state"]=="training" for r in state["reports"])>=2: renewed.set()
        return httpx.Response(200,json={"workerId":body["workerId"],"state":state["state"],"lastSeenAt":state["lastSeenAt"]})
    worker.connector.client._transport=httpx.MockTransport(handle)
    return worker,request,remote,state,renewed,calls


def test_worker_reports_availability_and_actual_inference(website):
    worker,_,_,_,_,calls=website
    assert worker.run_once()["state"]=="completed"
    states=[json.loads(r.content)["state"] for r in calls if r.url.path.endswith("/workers/heartbeat")]
    assert states==["ready","busy","ready"]


def test_training_presence_holds_execution_and_cannot_be_cleared_by_polling_worker(presence):
    worker,request,_,state,renewed,calls=presence
    outcome=[]
    with connected_training(worker.connector,worker_id=worker.worker_id,model_bundle_ids=worker.model_bundle_ids,heartbeat_seconds=.01) as check:
        assert state["state"]=="training"
        assert renewed.wait(timeout=2)
        check()
        # Same-thread CLI stages can reenter; an independent polling worker cannot.
        with model_execution(worker.connector.learning): pass
        thread=threading.Thread(target=lambda:outcome.append(worker.run_once()))
        thread.start();thread.join(timeout=2)
        assert outcome==[{"state":"busy","reason":"local_model_operation"}]
        assert state["state"]=="training"
        assert not any("/jobs/" in r.url.path for r in calls)
        assert worker.presence.report("ready")["state"]=="training"
    assert state["state"]=="ready" and state["operationId"] is None
    training=[r for r in state["reports"] if r["state"]=="training"]
    assert len({r["operationId"] for r in training})==1
    assert state["reports"][-1]["operationId"]==training[0]["operationId"]
    assert worker.run_once()["state"]=="completed"


def test_training_exception_releases_presence_and_execution(presence):
    worker,_,_,state,_,_=presence
    with pytest.raises(RuntimeError,match="synthetic optimizer failure"):
        with connected_training(worker.connector,worker_id=worker.worker_id,model_bundle_ids=worker.model_bundle_ids):
            raise RuntimeError("synthetic optimizer failure")
    assert state["state"]=="ready"
    with model_execution(worker.connector.learning): pass


def test_failed_cleanup_does_not_mask_training_failure_or_keep_execution_locked(presence):
    worker,_,_,state,_,_=presence
    state["readyUnavailable"]=True
    with pytest.raises(RuntimeError,match="synthetic optimizer failure"):
        with connected_training(worker.connector,worker_id=worker.worker_id,model_bundle_ids=worker.model_bundle_ids):
            raise RuntimeError("synthetic optimizer failure")
    assert state["state"]=="training"  # Server expiry clears the unrenewed report.
    assert state["reports"][-1]["state"]=="ready"
    with model_execution(worker.connector.learning): pass


def test_revoked_consent_is_mirrored_before_next_training_heartbeat(presence):
    worker,request,remote,state,renewed,_=presence
    with pytest.raises(PermissionError,match="paused"):
        with connected_training(worker.connector,worker_id=worker.worker_id,model_bundle_ids=worker.model_bundle_ids,heartbeat_seconds=.01) as check:
            remote["settings"]["mode"]="paused"
            remote["consents"]=[]
            until=time.monotonic()+2
            while time.monotonic()<until:
                with worker.connector.learning.transaction() as db:
                    if worker.connector._state(db)["mode"]=="paused": break
                time.sleep(.005)
            check()
    assert state["state"]=="ready"
    assert sum(r["state"]=="training" for r in state["reports"])==1
    with worker.connector.learning.transaction() as db:
        assert not worker.connector._state(db)["consents"]


def test_fresh_remote_training_never_causes_a_new_job_claim(presence):
    worker,_,_,state,_,calls=presence
    state.update(state="training",operationId="another-training-operation")
    assert worker.run_once()=={"state":"busy","reason":"training"}
    assert not any("/jobs/" in r.url.path for r in calls)


def test_busy_training_start_does_not_publish_false_presence(presence):
    worker,_,_,state,_,_=presence
    errors=[]
    def contend():
        try:
            with connected_training(worker.connector,worker_id=worker.worker_id,model_bundle_ids=worker.model_bundle_ids): pass
        except ExecutionBusy: errors.append("busy")
    with model_execution(worker.connector.learning):
        thread=threading.Thread(target=contend);thread.start();thread.join(timeout=2)
    assert errors==["busy"] and state["reports"]==[]


def test_actual_operator_call_keeps_cli_training_in_same_process_and_lock(tmp_path,monkeypatch):
    path=Path(__file__).parents[1]/"scripts"/"alpha_e2e_operator.py"
    spec=spec_from_file_location("alpha_operator_presence_test",path)
    operator=module_from_spec(spec);spec.loader.exec_module(operator)
    state_dir=tmp_path/"state"
    operator.call(state_dir,"init","--workspace","test-workspace")
    from argparse import Namespace
    from rateloop_evaluator import training
    _,_,store,_=cli.state(Namespace(state_dir=str(state_dir)))
    entered=[]
    def train(store,*args,**kwargs):
        with model_execution(store):
            entered.append(threading.get_ident())
            return {"modelDir":str(tmp_path/"candidate"),"training":{"optimizerSteps":1}}
    monkeypatch.setattr(training,"train_snapshot",train)
    with model_execution(store):
        result=operator.call(state_dir,"train","--snapshot-id","snapshot","--model-dir",tmp_path/"base",
            "--output",tmp_path/"candidate","--bundle-id","candidate","--max-steps","1")
    assert entered==[threading.get_ident()] and result["optimizerSteps"]==1


def test_presence_retries_only_classified_coordination_busy_with_identical_operation(website,monkeypatch):
    worker,*_=website
    requests=[]; delays=[]
    def handle(request):
        requests.append(json.loads(request.content))
        if len(requests)<3:
            return httpx.Response(503,json={"code":"database_coordination_busy","retryable":True})
        return httpx.Response(200,json={"workerId":worker.worker_id,"state":"training","lastSeenAt":iso(time.time())})
    worker.connector.client._transport=httpx.MockTransport(handle)
    monkeypatch.setattr("rateloop_evaluator.presence.time.sleep",delays.append)
    result=worker.presence.report("training",operation_id="training-test-operation")
    assert result["state"]=="training" and len(requests)==3
    assert requests==[requests[0]]*3 and delays==[.1,.3]


def test_presence_coordination_retry_is_bounded_and_does_not_expose_response(website,monkeypatch):
    worker,*_=website
    requests=[];delays=[]
    def handle(request):
        requests.append(request)
        return httpx.Response(503,json={"code":"database_coordination_busy","retryable":True,"message":"private-server-detail"})
    worker.connector.client._transport=httpx.MockTransport(handle)
    monkeypatch.setattr("rateloop_evaluator.presence.time.sleep",delays.append)
    with pytest.raises(ConnectorUnavailable) as raised:
        worker.presence.report("ready")
    assert raised.value.coordination_busy is True
    assert "private-server-detail" not in str(raised.value)
    assert len(requests)==3 and delays==[.1,.3]


@pytest.mark.parametrize("status,body",[
    (503,{"code":"different_failure","retryable":True}),
    (503,{"code":"database_coordination_busy","retryable":False}),
    (503,{"code":"database_coordination_busy","retryable":"true"}),
    (503,{"code":"database_coordination_busy","retryable":1}),
    (503,{"code":"database_coordination_busy"}),
    (503,{"code":"database_coordination_busy","retryable":True,"message":"x"*5000}),
    (503,"invalid response"),
    (503,b'{"code":"database_coordination_busy"'),
    (503,b'\xff'),
    (500,{"code":"database_coordination_busy","retryable":True}),
    (429,{"code":"database_coordination_busy","retryable":True}),
    (401,{"code":"database_coordination_busy","retryable":True}),
    (403,{"code":"database_coordination_busy","retryable":True}),
])
def test_presence_does_not_retry_unknown_failures_or_authentication(website,monkeypatch,status,body):
    worker,*_=website
    requests=[];delays=[]
    def handle(request):
        requests.append(request)
        return httpx.Response(status,content=body) if isinstance(body,bytes) else httpx.Response(status,json=body)
    worker.connector.client._transport=httpx.MockTransport(handle)
    monkeypatch.setattr("rateloop_evaluator.presence.time.sleep",delays.append)
    with pytest.raises(PermissionError if status in (401,403) else ConnectorUnavailable) as raised:
        worker.presence.report("ready")
    assert not getattr(raised.value,"coordination_busy",False)
    assert len(requests)==1 and delays==[]


def test_connector_writes_do_not_inherit_presence_retry(website):
    worker,*_=website
    requests=[]
    def handle(request):
        requests.append(request)
        return httpx.Response(503,json={"code":"database_coordination_busy","retryable":True})
    worker.connector.client._transport=httpx.MockTransport(handle)
    with pytest.raises(ConnectorUnavailable) as raised:
        worker.connector._request("POST","/receipts",json={})
    assert raised.value.coordination_busy is True and len(requests)==1
