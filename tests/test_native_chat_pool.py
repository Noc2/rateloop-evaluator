import json
import time

import httpx
import pytest

from test_hosted import config

from rateloop_evaluator.native_chat_pool import NativeChatPool, WORKER_ID, evaluate_native_job
from rateloop_evaluator.protocol import EvaluationRequest
from rateloop_evaluator.templates import CUSTOM_TEXT_CAPABILITY, custom_text_evaluation, custom_text_seed


class Backend:
    calls = 0
    length = 100
    def count_tokens(self, *_args): return self.length
    def predict(self, _text, questions):
        self.calls += 1
        return {question["id"]: {"approved": 0.8, "rejected": 0.2} for question in questions}


def fixture(language="en"):
    template = custom_text_evaluation(language,
        "Does this response address the user's request?" if language == "en" else "Geht diese Antwort auf die Anfrage ein?",
        "Yes" if language == "en" else "Ja", "No" if language == "en" else "Nein")
    request = EvaluationRequest.model_validate({"schemaVersion": "rateloop.evaluator.request.v1", "workspaceId": "workspace-a",
        "caseId": "case-a", "idempotencyKey": "native-case-a", "modelBundleId": "public-"+language,
        "template": template.model_dump(), "input": {"text": "A short relevant answer.", "context": "User request", "evidence": ""}, "deadlineMs": 5000})
    base = {"modelBundleId": request.modelBundleId, "templateCommitment": "sha256:"+"1"*64,
        "language": language, "taskCapability": CUSTOM_TEXT_CAPABILITY, "adapterCommitment": None, "trainingSnapshotCommitment": None}
    return {"workerId": WORKER_ID, "workspaceId": request.workspaceId, "jobId": "job-a", "modelBundleId": request.modelBundleId,
        "inputCommitment": request.input_commitment(), "templateCommitment": request.template_commitment(),
        "leaseToken": "a"*64, "leaseExpiresAt": __import__("datetime").datetime.fromtimestamp(time.time()+115,__import__("datetime").timezone.utc).isoformat(timespec="milliseconds").replace("+00:00","Z"),
        "agentId": "native-agent-a", "agentVersionId": "version-a", "baseRegistration": base,
        "content": {"agentId": "native-agent-a", "agentVersionId": "version-a", "request": request.model_dump(), "retainForTraining": False}}


def test_native_pool_reuses_real_shadow_core_for_both_locales_without_inventing_confidence(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    backend = Backend()
    for language in ("en", "de"):
        result = evaluate_native_job(fixture(language), backend=backend)
        assert result["outcome"] == "uncertain" and result["abstainReason"] == "uncalibrated"
        assert result["criteria"][0]["label"] == "approved"
        assert result["criteria"][0]["probabilities"] is None
        assert result["criteria"][0]["calibrationId"] is None
    assert backend.calls == 2
    assert not list(tmp_path.iterdir()), "Case grant, metadata and result cache must all be destroyed"


@pytest.mark.parametrize("change", [
    lambda job: job.update(workspaceId="workspace-b"),
    lambda job: job.update(workerId="external-worker"),
    lambda job: job.update(inputCommitment="sha256:"+"2"*64),
    lambda job: job["content"].update(agentVersionId="external-version"),
    lambda job: job["content"].update(retainForTraining=True),
    lambda job: job["baseRegistration"].update(adapterCommitment="sha256:"+"3"*64),
    lambda job: job.update(leaseExpiresAt="2000-01-01T00:00:00.000Z"),
])
def test_invalid_or_expired_exact_case_grants_do_not_infer(change):
    job = fixture(); backend = Backend(); change(job)
    with pytest.raises((ValueError, PermissionError)): evaluate_native_job(job, backend=backend)
    assert backend.calls == 0


def test_over_limit_input_abstains_without_model_prediction():
    backend = Backend(); backend.length = 513
    result = evaluate_native_job(fixture(), backend=backend)
    assert result["abstainReason"] == "input_too_long" and not result["criteria"] and backend.calls == 0


def test_native_worker_checks_permission_before_inference_and_retries_receipt_without_scoring_again():
    job = fixture(); backend = Backend(); events = []; completions = []
    def transport(request):
        assert request.url == "https://www.rateloop.ai/api/internal/native-chat-evaluator"
        body = json.loads(request.content); events.append(body["action"])
        if body["action"] == "claim": return httpx.Response(200, json={"job": job})
        if body["action"] == "heartbeat_job": return httpx.Response(200, json={"leaseExpiresAt": job["leaseExpiresAt"]})
        if body["action"] == "complete":
            completions.append(body["result"])
            return httpx.Response(503 if len(completions) == 1 else 200, json={"completed": True})
        raise AssertionError("Unexpected action")
    pool = NativeChatPool(secret="s"*32, base_url="https://www.rateloop.ai", bundles=[job["baseRegistration"]], backend=backend,
        transport=httpx.MockTransport(transport))
    from rateloop_evaluator.connector import ConnectorUnavailable
    with pytest.raises(ConnectorUnavailable): pool.run_once()
    assert pool.run_once()["state"] == "completed"
    assert backend.calls == 1 and completions[0] == completions[1]
    assert events == ["claim", "heartbeat_job", "complete", "complete"]
    pool.close()


@pytest.mark.parametrize("status", [400,413,422])
def test_permanently_rejected_receipt_fails_its_lease_and_does_not_block_other_native_jobs(status):
    job=fixture(); backend=Backend(); events=[]; claimed=[False]
    def transport(request):
        body=json.loads(request.content); action=body["action"]; events.append(action)
        if action=="claim":
            value=None if claimed[0] else job; claimed[0]=True
            return httpx.Response(200,json={"job":value})
        if action=="heartbeat_job": return httpx.Response(200,json={"leaseExpiresAt":job["leaseExpiresAt"]})
        if action=="complete": return httpx.Response(status,json={"code":"invalid_receipt"})
        if action=="fail":
            assert body=={"action":"fail","workspaceId":job["workspaceId"],"jobId":job["jobId"],
                "leaseToken":job["leaseToken"],"retryable":False,"errorCode":"native_receipt_rejected"}
            return httpx.Response(200,json={"retrying":False})
        raise AssertionError(action)
    pool=NativeChatPool(secret="s"*32,base_url="https://www.rateloop.ai",bundles=[job["baseRegistration"]],backend=backend,
        transport=httpx.MockTransport(transport))
    assert pool.run_once()=={"state":"failed"}
    assert pool.run_once()=={"state":"idle"}
    assert pool.pending is None and backend.calls==1
    assert events==["claim","heartbeat_job","complete","fail","claim"]
    pool.close()


def test_transient_receipt_failure_retains_only_fencing_and_result_until_original_lease_expires(monkeypatch):
    from rateloop_evaluator.connector import ConnectorUnavailable, _timestamp
    job=fixture(); backend=Backend(); events=[]; now=[time.time()]; claimed=[False]
    monkeypatch.setattr("rateloop_evaluator.native_chat_pool.time.time",lambda:now[0])
    def transport(request):
        action=json.loads(request.content)["action"]; events.append(action)
        if action=="claim":
            value=None if claimed[0] else job; claimed[0]=True
            return httpx.Response(200,json={"job":value})
        if action=="heartbeat_job": return httpx.Response(200,json={"leaseExpiresAt":job["leaseExpiresAt"]})
        if action=="complete": return httpx.Response(503,json={"code":"unavailable"})
        raise AssertionError(action)
    pool=NativeChatPool(secret="s"*32,base_url="https://www.rateloop.ai",bundles=[job["baseRegistration"]],backend=backend,
        transport=httpx.MockTransport(transport))
    with pytest.raises(ConnectorUnavailable): pool.run_once()
    assert pool.pending is not None and job["content"]["request"]["input"]["text"] not in json.dumps(pool.pending)
    now[0]=_timestamp(job["leaseExpiresAt"])
    assert pool.run_once()=={"state":"lease_lost"}
    assert pool.pending is None and pool.run_once()=={"state":"idle"}
    assert backend.calls==1 and events==["claim","heartbeat_job","complete","claim"]
    pool.close()


def test_rejected_receipt_retries_only_content_free_terminal_failure_within_its_lease():
    from rateloop_evaluator.connector import ConnectorUnavailable
    job=fixture(); backend=Backend(); events=[]; failures=[]
    def transport(request):
        body=json.loads(request.content); action=body["action"]; events.append(action)
        if action=="claim": return httpx.Response(200,json={"job":job})
        if action=="heartbeat_job": return httpx.Response(200,json={"leaseExpiresAt":job["leaseExpiresAt"]})
        if action=="complete": return httpx.Response(422,json={"code":"invalid_receipt"})
        if action=="fail":
            failures.append(body)
            return httpx.Response(503 if len(failures)==1 else 200,json={"retrying":False})
        raise AssertionError(action)
    pool=NativeChatPool(secret="s"*32,base_url="https://www.rateloop.ai",bundles=[job["baseRegistration"]],backend=backend,
        transport=httpx.MockTransport(transport))
    with pytest.raises(ConnectorUnavailable): pool.run_once()
    assert pool.pending is not None and pool.pending[1] is None
    assert pool.run_once()=={"state":"failed"}
    assert failures[0]==failures[1] and backend.calls==1 and pool.pending is None
    assert events==["claim","heartbeat_job","complete","fail","fail"]
    pool.close()


@pytest.mark.parametrize("operation",["count_tokens","predict"])
def test_core_inference_exceptions_fail_promptly_without_retaining_or_disclosing_input(operation,tmp_path,monkeypatch,capsys):
    monkeypatch.setattr("tempfile.tempdir",str(tmp_path))
    job=fixture(); backend=Backend(); events=[]; claimed=[False]
    def failed_inference(*args): raise RuntimeError(job["content"]["request"]["input"]["text"])
    setattr(backend,operation,failed_inference)
    def transport(request):
        body=json.loads(request.content); action=body["action"]; events.append(action)
        if action=="claim":
            value=None if claimed[0] else job; claimed[0]=True
            return httpx.Response(200,json={"job":value})
        if action=="heartbeat_job": return httpx.Response(200,json={"leaseExpiresAt":job["leaseExpiresAt"]})
        if action=="fail":
            assert body=={"action":"fail","workspaceId":job["workspaceId"],"jobId":job["jobId"],
                "leaseToken":job["leaseToken"],"retryable":False,"errorCode":"native_inference_failed"}
            return httpx.Response(200,json={"retrying":False})
        raise AssertionError(action)
    pool=NativeChatPool(secret="s"*32,base_url="https://www.rateloop.ai",bundles=[job["baseRegistration"]],backend=backend,
        transport=httpx.MockTransport(transport))
    assert pool.run_once()=={"state":"failed"}
    assert pool.pending is None and pool.run_once()=={"state":"idle"}
    assert events==["claim","heartbeat_job","fail","claim"] and not list(tmp_path.iterdir())
    captured=capsys.readouterr()
    assert not captured.out and not captured.err
    pool.close()


def test_core_failure_retries_only_failure_metadata_when_reporting_is_temporarily_unavailable():
    from rateloop_evaluator.connector import ConnectorUnavailable
    job=fixture(); backend=Backend(); events=[]; failures=[]; predictions=[0]
    def failed_inference(*args): predictions[0]+=1; raise RuntimeError("private inference detail")
    backend.predict=failed_inference
    def transport(request):
        body=json.loads(request.content); action=body["action"]; events.append(action)
        if action=="claim": return httpx.Response(200,json={"job":job})
        if action=="heartbeat_job": return httpx.Response(200,json={"leaseExpiresAt":job["leaseExpiresAt"]})
        if action=="fail":
            failures.append(body)
            return httpx.Response(503 if len(failures)==1 else 200,json={"retrying":False})
        raise AssertionError(action)
    pool=NativeChatPool(secret="s"*32,base_url="https://www.rateloop.ai",bundles=[job["baseRegistration"]],backend=backend,
        transport=httpx.MockTransport(transport))
    with pytest.raises(ConnectorUnavailable): pool.run_once()
    assert pool.pending is not None and pool.pending[1] is None
    assert "private inference detail" not in json.dumps(pool.pending)
    assert job["content"]["request"]["input"]["text"] not in json.dumps(pool.pending)
    assert pool.run_once()=={"state":"failed"}
    assert failures[0]==failures[1] and predictions==[1] and pool.pending is None
    assert events==["claim","heartbeat_job","fail","fail"]
    pool.close()


def test_grant_expiring_during_inference_releases_no_receipt_or_private_case_files(tmp_path,monkeypatch):
    from rateloop_evaluator.connector import _timestamp
    monkeypatch.setattr("tempfile.tempdir",str(tmp_path))
    job=fixture(); backend=Backend(); events=[]; now=[time.time()]
    monkeypatch.setattr("rateloop_evaluator.native_chat_pool.time.time",lambda:now[0])
    original_predict=backend.predict
    def expire_during_prediction(*args):
        result=original_predict(*args)
        now[0]=_timestamp(job["leaseExpiresAt"])
        return result
    backend.predict=expire_during_prediction
    def transport(request):
        action=json.loads(request.content)["action"]; events.append(action)
        if action=="claim": return httpx.Response(200,json={"job":job})
        if action=="heartbeat_job": return httpx.Response(200,json={"leaseExpiresAt":job["leaseExpiresAt"]})
        raise AssertionError("Expired grant must release no completion or failure content")
    pool=NativeChatPool(secret="s"*32,base_url="https://www.rateloop.ai",bundles=[job["baseRegistration"]],backend=backend,
        transport=httpx.MockTransport(transport))
    assert pool.run_once()=={"state":"lease_lost"}
    assert backend.calls==1 and pool.pending is None and events==["claim","heartbeat_job"]
    assert not list(tmp_path.iterdir())
    pool.close()


def test_revocation_before_inference_does_not_score():
    job = fixture(); backend = Backend()
    def transport(request):
        action = json.loads(request.content)["action"]
        return httpx.Response(200, json={"job": job}) if action == "claim" else httpx.Response(403, json={"code": "permission_withdrawn"})
    pool = NativeChatPool(secret="s"*32, base_url="https://www.rateloop.ai", bundles=[job["baseRegistration"]], backend=backend,
        transport=httpx.MockTransport(transport))
    assert pool.run_once()["state"] == "lease_lost"
    assert backend.calls == 0
    pool.close()


def test_native_pool_configuration_is_distinct_strict_and_never_rebinds_the_retained_connector(tmp_path):
    from copy import deepcopy
    from rateloop_evaluator import cli, hosted
    from test_cli import model_files
    config = {"schemaVersion":"rateloop.hosted-worker.v1","workspaceId":"retained-workspace","workerId":"retained-worker",
        "modelDir":str(model_files(tmp_path)),"stateDir":str(tmp_path/"state"),
        "bundles":[{"language":"en","modelBundleId":"retained-en"},{"language":"de","modelBundleId":"retained-de"}],
        "connection":{"baseUrl":"https://www.rateloop.ai","apiKey":"retained-private-key","apiKeyId":"retained-key",
            "agentId":"retained-agent","agentVersionId":"retained-version","metadataUploadEnabled":True},"pollSeconds":5,"healthPort":8080,
        "nativeChatPool":{"baseUrl":"https://www.rateloop.ai","secret":"s"*32}}
    path = tmp_path/"config.json"; cli.write_private(path,config)
    assert hosted.read_config(path) == config
    hosted.bootstrap(config)
    changed = deepcopy(config); changed["nativeChatPool"]["secret"] = "r"*32
    assert hosted.bootstrap(changed) == hosted.bootstrap(config)
    for alter in (lambda pool:pool.update(baseUrl="https://external.example"), lambda pool:pool.update(secret="short"),
                  lambda pool:pool.update(secret="x"*31+"\n"), lambda pool:pool.update(extra="unknown")):
        invalid = deepcopy(config); alter(invalid["nativeChatPool"]); cli.write_private(path,invalid)
        with pytest.raises(ValueError): hosted.read_config(path)


def test_native_pool_unavailable_does_not_revoke_retained_worker_health_and_closes_both_queues(tmp_path, monkeypatch):
    from contextlib import nullcontext
    import signal
    import sys
    from types import SimpleNamespace
    from rateloop_evaluator import hosted
    from test_cli import model_files
    from rateloop_evaluator.connector import ConnectorUnavailable
    config = {"schemaVersion":"rateloop.hosted-worker.v1","workspaceId":"retained-workspace","workerId":"retained-worker",
        "modelDir":str(model_files(tmp_path)),"stateDir":str(tmp_path/"state"),
        "bundles":[{"language":language,"modelBundleId":"custom-"+language,"taskCapability":CUSTOM_TEXT_CAPABILITY} for language in ("en","de")],
        "connection":{"baseUrl":"https://www.rateloop.ai","apiKey":"retained-private-key","apiKeyId":"retained-key",
            "agentId":"retained-agent","agentVersionId":"retained-version","metadataUploadEnabled":True},"pollSeconds":5,"healthPort":8080,
        "nativeChatPool":{"baseUrl":"https://www.rateloop.ai","secret":"s"*32}}
    events = []; state = {}
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(set_num_threads=lambda n:None,set_num_interop_threads=lambda n:None))
    monkeypatch.setattr(hosted,"health_server",lambda health,_port:state.update(health=health) or nullcontext())
    monkeypatch.setattr("rateloop_evaluator.execution.model_execution",lambda learning:nullcontext())
    class Connector:
        learning = object()
        def __init__(self,**kwargs): events.append("old-connected")
        def close(self): events.append("old-closed")
    class Worker:
        def __init__(self,*args,evaluate,**kwargs): self.evaluate=evaluate; self.last_label_sync=time.monotonic()
        def run_once(self, *, include_training=True):
            assert include_training is False
            events.append("old-poll")
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM,None)
            return {"state":"idle"}
    class Pool:
        def __init__(self,**kwargs): events.append("native-connected")
        def run_once(self): events.append("native-poll"); raise ConnectorUnavailable("queue unavailable")
        def close(self): events.append("native-closed")
    def evaluate(): pass
    evaluate.native_backend = Backend()
    evaluate.close = lambda:events.append("model-closed")
    monkeypatch.setattr(hosted,"RateLoopConnector",Connector)
    monkeypatch.setattr(hosted,"prepare_evaluator",lambda *_args,**kwargs:evaluate)
    monkeypatch.setattr(hosted,"OutboundWorker",Worker)
    monkeypatch.setattr("rateloop_evaluator.native_chat_pool.NativeChatPool",Pool)
    monkeypatch.setattr("rateloop_evaluator.native_chat_pool.validate_registrations",lambda bundles,_path,**_kwargs:bundles)
    hosted.run_hosted(config)
    assert events == ["old-connected","native-connected","old-poll","native-poll","native-closed","model-closed","old-closed"]
    assert state["health"].last_success is not None


def test_native_evidence_is_opt_in_content_free_and_retried_without_reinference():
    from rateloop_evaluator.connector import ConnectorUnavailable
    job=fixture();job['evidenceVersion']='rateloop.evaluator.evidence.v2'
    backend=Backend(); completions=[]
    def transport(request):
        body=json.loads(request.content)
        if body['action']=='claim': return httpx.Response(200,json={'job':job})
        if body['action']=='heartbeat_job': return httpx.Response(200,json={'leaseExpiresAt':job['leaseExpiresAt']})
        if body['action']=='complete':
            completions.append(body)
            return httpx.Response(503 if len(completions)==1 else 200,json={'completed':True})
        raise AssertionError(body['action'])
    pool=NativeChatPool(secret='s'*32,base_url='https://www.rateloop.ai',bundles=[job['baseRegistration']],backend=backend,
        transport=httpx.MockTransport(transport))
    with pytest.raises(ConnectorUnavailable):pool.run_once()
    assert pool.run_once()=={'state':'completed'}
    assert completions[0]==completions[1] and backend.calls==1
    evidence=completions[0]['evidence']
    assert evidence['checks'][0]['judgment']=='meets'
    assert evidence['checks'][-1]['state']=='not_checked'
    assert job['content']['request']['input']['text'] not in json.dumps(completions)
    pool.close()


def test_native_evidence_unknown_version_never_infers():
    job=fixture();job['evidenceVersion']='unknown';backend=Backend()
    with pytest.raises(ValueError):evaluate_native_job(job,backend=backend)
    assert backend.calls==0


def test_native_v2_runtime_identity_matches_real_registration_export(config):
    from pathlib import Path
    from rateloop_evaluator.hosted import bootstrap,validate_pinned_model
    from rateloop_evaluator.native_chat_pool import validate_registrations
    config['bundles']=[{'language':language,'modelBundleId':'native-'+language,'taskCapability':CUSTOM_TEXT_CAPABILITY} for language in ('en','de')]
    output=bootstrap(config)
    registrations=[json.loads(Path(path).read_text()) for path in output['registrations']]
    assert validate_registrations(registrations,config['modelDir'])==registrations
    backend=Backend();backend.question_execution='joint_schema'
    backend.manifest=validate_pinned_model(Path(config['modelDir']))
    for registration in registrations:
        job=fixture(registration['language']);job['baseRegistration']=registration
        job['evidenceVersion']='rateloop.evaluator.evidence.v2'
        request=EvaluationRequest.model_validate(job['content']['request'])
        request.modelBundleId=registration['modelBundleId']
        job.update(modelBundleId=request.modelBundleId,inputCommitment=request.input_commitment())
        job['content']['request']=request.model_dump()
        evidence=evaluate_native_job(job,backend=backend)['evidence']
        identity=evidence['identity']
        assert identity['weightsCommitment']==registration['baseWeightsCommitment']
        assert identity['tokenizerCommitment']==registration['tokenizerCommitment']
        assert identity['scoreAdapter']==registration['scoreCapability']['adapter']
        assert identity['precision']==registration['quantization']=='fp32'
