"""Outbound, fenced website-job worker. No listener, arbitrary downloads or commands."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import fcntl
import hashlib
import os
from pathlib import Path
import signal
import threading
import time
from typing import Callable

from fastapi import HTTPException

from .connector import ConnectorRejected, ConnectorUnavailable, RateLoopConnector, _hash, _opaque, _timestamp
from .protocol import EvaluationRequest, EvaluationResult
from .templates import overall_approval
from .execution import ExecutionBusy, model_execution


class OutboundWorker:
    def __init__(self, connector: RateLoopConnector, *, worker_id: str, model_bundle_ids: list[str],
                 evaluate: Callable[[EvaluationRequest],dict], poll_seconds: float = 5,
                 heartbeat_seconds: float = 30):
        _opaque(worker_id)
        if not model_bundle_ids or len(set(model_bundle_ids)) != len(model_bundle_ids):
            raise ValueError("Worker requires explicit unique pinned model bundles")
        for bundle in model_bundle_ids: _opaque(bundle)
        if not 1 <= poll_seconds <= 60 or not 1 <= heartbeat_seconds <= 30:
            raise ValueError("Polling must be 1-60 seconds and heartbeat 1-30 seconds")
        if not connector.metadata_upload_enabled:
            raise PermissionError("Worker requires explicit receipt upload permission")
        self.connector=connector; self.worker_id=worker_id; self.model_bundle_ids=model_bundle_ids
        self.evaluate=evaluate; self.poll_seconds=poll_seconds; self.heartbeat_seconds=heartbeat_seconds
        self.stop=threading.Event()
        self.last_label_sync=0.0

    def sync_labels(self) -> dict:
        """Import only the website's exact overall-question labels under explicit learning consent."""
        from .protocol import commitment
        with self.connector.learning.transaction() as database:
            consents=deepcopy(self.connector._state(database).get("consents",{}))
        imported=0; rejected=0; truncated=False
        for consent_id,record in consents.items():
            consent=record["consent"]
            if consent["purpose"] != "private_learning" or "human_labels" not in consent["fields"]: continue
            for language in ("en","de"):
                template_digest=commitment(overall_approval(language).model_dump(),"rateloop.evaluator.template.v1")
                if template_digest not in consent["templateCommitments"]: continue
                report=self.connector.fetch_and_import_labels(consent_id,question_id="overall_approval",template_commitment=template_digest,
                    outcome_labels={"positive":"approved","negative":"rejected"})
                imported+=report["imported"]; rejected+=len(report["rejected"]); truncated=truncated or report["truncated"]
        self.last_label_sync=time.monotonic()
        return {"imported":imported,"rejected":rejected,"truncated":truncated}

    def _remember_collection(self, request: EvaluationRequest, body: dict) -> None:
        created_at=_timestamp(body.get("createdAt"))
        if not 0<created_at<=time.time()+300 or type(body.get("retainForTraining")) is not bool:
            raise ValueError("Website content must declare its original collection and training permission")
        fields={"input"}
        if request.input.context: fields.add("context")
        if request.input.evidence: fields.add("evidence")
        with self.connector.learning.transaction() as database:
            state=self.connector._state(database)
            permitted=any(c["consent"]["purpose"]=="private_learning" and _timestamp(c["consent"]["issuedAt"])<=created_at
                and request.modelBundleId in c["consent"]["modelBundleIds"]
                and request.template_commitment() in c["consent"]["templateCommitments"]
                and fields<=set(c["consent"]["fields"]) for c in state.get("consents",{}).values())
            value={"caseId":request.caseId,"createdAt":body["createdAt"],"trainingAllowed":body["retainForTraining"] and permitted}
            collections=state.setdefault("collections",{})
            old=collections.get(request.input_commitment())
            if old and (old["createdAt"]!=value["createdAt"] or old["caseId"]!=request.caseId):
                raise ValueError("Website collection identity changed")
            if old: value["trainingAllowed"]=old["trainingAllowed"] and value["trainingAllowed"]
            collections[request.input_commitment()]=value

    def _saved(self) -> dict | None:
        with self.connector.learning.transaction() as database:
            return deepcopy(self.connector._state(database).get("worker_jobs",{}).get(self.worker_id))

    def _save(self, job: dict | None) -> None:
        with self.connector.learning.transaction() as database:
            jobs=self.connector._state(database).setdefault("worker_jobs",{})
            if job is None: jobs.pop(self.worker_id,None)
            else: jobs[self.worker_id]=deepcopy(job)

    def _post(self, job: dict, action: str, **extra) -> dict:
        return self.connector._request("POST",f"/jobs/{job['jobId']}/{action}",json={
            "workerId":self.worker_id,"leaseToken":job["leaseToken"],**extra})

    def _heartbeat(self, job: dict) -> None:
        response=self._post(job,"heartbeat")
        until=_timestamp(response.get("leaseExpiresAt"))
        if not time.time() < until <= time.time()+125:
            raise ValueError("Invalid renewed job lease")
        job["leaseExpiresAt"]=response["leaseExpiresAt"]
        self._save(job)

    def _validate_claim(self, job: dict) -> None:
        for key in ("jobId","modelBundleId"): _opaque(job.get(key))
        for key in ("inputCommitment","templateCommitment"): _hash(job.get(key))
        if job["modelBundleId"] not in self.model_bundle_ids:
            raise PermissionError("Server job requested an unconfigured model")
        token=job.get("leaseToken")
        if not isinstance(token,str) or not 16 <= len(token) <= 512 or any(c in token for c in "\r\n"):
            raise ValueError("Invalid job fencing token")
        if not time.time() < _timestamp(job.get("leaseExpiresAt")) <= time.time()+125:
            raise ValueError("Job lease is expired or exceeds 120 seconds")

    def _remember_audit(self, request: EvaluationRequest, body: dict) -> None:
        audit=body.get("audit",{})
        if (audit.get("selected") is not True or audit.get("kind") != "mandatory" or audit.get("aiExposed") is not False
                or audit.get("selectionProbabilityBps") != 10000 or audit.get("blindingAssurance") != "server_enforced"):
            raise PermissionError("Website jobs require a mandatory server-blinded human review")
        _opaque(audit.get("auditId"))
        selected_at=_timestamp(audit.get("selectedAt"))
        if not 0 < selected_at <= time.time()+300:
            raise ValueError("Invalid pre-scoring audit timestamp")
        bindings={key:_hash(audit.get(key)) for key in ("sourceContentHash","suggestedContentHash","frozenQuestionHash")}
        if (bindings["sourceContentHash"] != "sha256:"+hashlib.sha256(request.input.context.encode()).hexdigest()
                or bindings["suggestedContentHash"] != "sha256:"+hashlib.sha256(request.input.text.encode()).hexdigest()):
            raise ValueError("Human review and evaluator content bytes differ")
        if request.template != overall_approval(request.template.language):
            raise ValueError("Website jobs require the exact frozen overall approval template")
        with self.connector.learning.transaction() as database:
            self.connector._case_live(database,request.caseId)
            state=self.connector._state(database)
            previous=state["audits"].get(request.input_commitment())
            if previous and previous.get("response",{}).get("auditId") != audit["auditId"]:
                raise ValueError("Resumed job changed the independent review")
            state["audits"][request.input_commitment()]={"response":deepcopy(audit),"case_id":request.caseId,
                "model_bundle_id":request.modelBundleId,"template_commitment":request.template_commitment(),
                "selected_before_scoring":True,"selected_at":selected_at,"ai_exposed":False,**bindings}

    @contextmanager
    def _renew_while_working(self, job: dict):
        stopped=threading.Event(); failed=[]
        def renew():
            while not stopped.wait(self.heartbeat_seconds):
                try:
                    status=self.connector.sync_grants()
                    if status["mode"] != "shadow": raise PermissionError("Workspace paused")
                    self._heartbeat(job)
                except Exception as error:
                    failed.append(error); return
        thread=threading.Thread(target=renew,name="evaluator-lease",daemon=True); thread.start()
        try:
            yield failed
        finally:
            stopped.set(); thread.join(timeout=35)
            if thread.is_alive(): failed.append(ConnectorUnavailable("Lease renewal is still unavailable"))

    def _process(self, job: dict) -> dict:
        body=self.connector._request("GET",f"/jobs/{job['jobId']}/content",headers={
            "X-Evaluator-Lease":job["leaseToken"],"X-Evaluator-Worker":self.worker_id})
        if (body.get("agentId"),body.get("agentVersionId")) != (self.connector.agent_id,self.connector.agent_version_id):
            raise PermissionError("Worker identity does not match the submitted agent version")
        request=EvaluationRequest.model_validate(body.get("request"))
        if (request.workspaceId != self.connector.workspace_id or request.modelBundleId != job["modelBundleId"]
                or request.input_commitment()!=job["inputCommitment"] or request.template_commitment()!=job["templateCommitment"]):
            raise ValueError("Job content does not match its committed workspace, template and model")
        self._remember_audit(request,body)
        self._remember_collection(request,body)
        job["caseId"]=request.caseId; self._save(job)
        with self._renew_while_working(job) as failed:
            result=EvaluationResult.model_validate(self.evaluate(request))
        if failed: raise failed[0]
        if (result.workspaceId,result.caseId,result.modelBundleId,result.inputCommitment,result.templateCommitment) != (
                request.workspaceId,request.caseId,request.modelBundleId,request.input_commitment(),request.template_commitment()):
            raise ValueError("Evaluation changed the committed website case")
        # Fresh checks after a slow model load/inference and before any result leaves.
        if self.connector.sync_grants()["mode"] != "shadow": raise PermissionError("Workspace paused")
        self._heartbeat(job)
        with self.connector.learning.transaction() as database:
            self.connector._case_live(database,request.caseId)
            self.connector._state(database)["results"][result.inputCommitment]=result.model_dump()
        receipt_key=self.connector.queue_result(result,job_context={"jobId":job["jobId"],"workerId":self.worker_id,"leaseToken":job["leaseToken"]})
        acknowledgment=self.connector.runtime.acknowledgment(receipt_key)
        if acknowledgment is None:
            self.connector.flush()
            acknowledgment=self.connector.runtime.acknowledgment(receipt_key)
        if acknowledgment is None:
            raise ConnectorUnavailable("Result remains in the encrypted outbox")
        self._post(job,"complete",receiptId=acknowledgment["receiptId"])
        self._save(None)
        return {"state":"completed","jobId":job["jobId"],"modelBundleId":job["modelBundleId"],"humanReviewRequired":True}

    def run_once(self) -> dict:
        status=self.connector.sync_grants()
        if status["mode"] != "shadow":
            return {"state":"paused"}
        try:
            with model_execution(self.connector.learning):
                return self._run_available()
        except ExecutionBusy:
            return {"state":"busy","reason":"local_model_operation"}

    def _run_available(self) -> dict:
        job=self._saved()
        if job:
            try:
                self._heartbeat(job)
            except ConnectorRejected as error:
                if error.status not in (404,409,410): raise
                self._save(None); job=None
        if job is None:
            response=self.connector._request("POST","/jobs/claim",json={"workerId":self.worker_id,"modelBundleIds":self.model_bundle_ids})
            job=response.get("job")
            if job is None: return {"state":"idle"}
            self._validate_claim(job); self._save(job)
        try:
            return self._process(job)
        except ConnectorUnavailable:
            raise  # Keep the fencing token for recovery; never duplicate a review.
        except ConnectorRejected as error:
            if error.status in (404,409,410):
                self._save(None)
                return {"state":"lease_lost","jobId":job["jobId"]}
            raise
        except (ValueError,PermissionError,HTTPException):
            try: self._post(job,"fail",retryable=False,errorCode="local_validation_failed")
            except (ConnectorUnavailable,ConnectorRejected,PermissionError): pass
            self._save(None)
            return {"state":"failed","jobId":job["jobId"],"errorCode":"local_validation_failed"}

    def run(self) -> None:
        failures=0
        while not self.stop.is_set():
            try:
                self.run_once(); failures=0
                if time.monotonic()-self.last_label_sync>=60: self.sync_labels()
            except (ConnectorUnavailable,ConnectorRejected,PermissionError,ValueError):
                failures+=1
            self.stop.wait(min(60,self.poll_seconds*2**min(failures,5)))


@contextmanager
def single_worker(root: Path, worker_id: str):
    """The same worker identity must never run twice against one local state."""
    digest=hashlib.sha256(worker_id.encode()).hexdigest()
    path=root / ("worker-"+digest+".lock")
    fd=os.open(path,os.O_CREAT|os.O_RDWR|getattr(os,"O_NOFOLLOW",0),0o600)
    try:
        try: fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError("This worker is already running") from None
        yield
    finally:
        os.close(fd)


def run_worker(worker: OutboundWorker, root: Path, *, once: bool = False):
    with single_worker(root,worker.worker_id):
        if once: return worker.run_once()
        previous={signum:signal.signal(signum,lambda *_:worker.stop.set()) for signum in (signal.SIGINT,signal.SIGTERM)}
        try: worker.run()
        finally:
            for signum,handler in previous.items(): signal.signal(signum,handler)
    return {"state":"stopped"}


def install_launchd(*, state_dir: Path, config_path: str, worker_id: str, bundle_ids: list[str], device: str,
                    poll_seconds: float, output: str, load: bool = False) -> dict:
    """Write an owner-only native Mac service; credentials stay in their private file."""
    import plistlib
    import subprocess
    import sys
    from .learning import read_secret
    if sys.platform != "darwin": raise ValueError("launchd installation requires macOS")
    _opaque(worker_id)
    for bundle in bundle_ids: _opaque(bundle)
    if not bundle_ids or device not in ("cpu","mps","cuda") or not 1<=poll_seconds<=60:
        raise ValueError("Invalid launchd worker configuration")
    config=Path(config_path).expanduser().resolve(); read_secret(config)
    destination=Path(output).expanduser().absolute()
    destination.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    if destination.is_symlink(): raise ValueError("Launch agent cannot be a symlink")
    arguments=[sys.executable,"-m","rateloop_evaluator.cli","--state-dir",str(state_dir.resolve()),"worker",
        "--config",str(config),"--worker-id",worker_id,"--device",device,"--poll-seconds",str(poll_seconds)]
    for bundle in bundle_ids: arguments.extend(["--bundle-id",bundle])
    label="ai.rateloop.evaluator."+hashlib.sha256(worker_id.encode()).hexdigest()[:16]
    value={"Label":label,"ProgramArguments":arguments,"WorkingDirectory":str(state_dir.resolve()),
        "RunAtLoad":True,"KeepAlive":True,"ThrottleInterval":30,"ProcessType":"Background",
        "StandardOutPath":"/dev/null","StandardErrorPath":"/dev/null",
        "EnvironmentVariables":{"PYTHONUNBUFFERED":"1","HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"}}
    descriptor=os.open(destination,os.O_CREAT|os.O_EXCL|os.O_WRONLY|getattr(os,"O_NOFOLLOW",0),0o600)
    with os.fdopen(descriptor,"wb") as stream: plistlib.dump(value,stream)
    if load:
        subprocess.run(["launchctl","bootstrap",f"gui/{os.getuid()}",str(destination)],check=True,capture_output=True)
    return {"launchAgent":str(destination),"label":label,"loaded":load,"outboundOnly":True}
