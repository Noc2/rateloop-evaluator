"""Customer-owned outbound generation with fenced, encrypted completion retries."""
from __future__ import annotations

from contextlib import contextmanager
import json
import threading
import time

import httpx

from .connector import bounded_response_object, _opaque, _timestamp, ConnectorUnavailable, ConnectorRejected
from .enrollment import enrollment_origin
from .ollama import OllamaRuntime, OllamaError
from .protocol import commitment

_API = "/api/assurance/v2/generation"


class GenerationWorker:
    def __init__(self, *, connection: dict, worker_id: str, model: dict, runtime: OllamaRuntime, outbox,
                 heartbeat_seconds=1, transport=None):
        self.worker_id=_opaque(worker_id);self.model=model;self.runtime=runtime;self.outbox=outbox
        self.model_commitment=commitment(model,"rateloop.generation-model.v1")
        if not 1<=heartbeat_seconds<=30: raise ValueError("Generation heartbeat must be 1-30 seconds")
        self.heartbeat_seconds=heartbeat_seconds
        self.stop=threading.Event()
        self.client=httpx.Client(base_url=enrollment_origin(connection["baseUrl"],allow_insecure_loopback=connection.get("allowInsecureLoopback",False)),
            headers={"Authorization":"Bearer "+connection["apiKey"]},timeout=15,trust_env=False,follow_redirects=False,transport=transport)
        self.registered=False

    def close(self): self.client.close();self.runtime.close()

    def _post(self, path, body):
        try:
            with self.client.stream("POST",_API+path,json=body) as response:
                if response.status_code in (401,403): raise PermissionError("Generation device is no longer authorized")
                if response.status_code>=500 or response.status_code==429:
                    raise ConnectorUnavailable("RateLoop generation service is unavailable")
                if not 200<=response.status_code<300: raise ConnectorRejected(response.status_code)
                return bounded_response_object(response,max_bytes=300_000)
        except httpx.HTTPError: raise ConnectorUnavailable("RateLoop generation service is unavailable") from None

    def _action(self, job, action, **body):
        return self._post("/jobs/"+job["jobId"]+"/"+action,{"workerId":self.worker_id,"leaseToken":job["leaseToken"],**body})

    def register(self):
        if self.runtime.identity()!=self.model: raise OllamaError("model_unavailable")
        result=self._post("/models",{"workerId":self.worker_id,"model":self.model})
        if result.get("modelCommitment")!=self.model_commitment: raise ValueError("Generation model registration changed identity")
        self.registered=True

    def _validate_job(self, job):
        if not isinstance(job,dict) or set(job)!={"jobId","leaseToken","request"}: raise ValueError("Invalid generation lease")
        _opaque(job["jobId"])
        token=job['leaseToken']
        if not isinstance(token,str) or not 20<=len(token)<=256 or any(c in token for c in '\r\n'):
            raise ValueError("Invalid generation lease")
        request=job["request"]
        if (not isinstance(request,dict) or set(request)!={"schemaVersion","jobId","model","modelCommitment","messages",
                "maxOutputTokens","maxOutputCharacters","deadline"} or request["schemaVersion"]!="rateloop.generation-request.v1"
                or request["jobId"]!=job["jobId"] or request["model"]!=self.model or request["modelCommitment"]!=self.model_commitment):
            raise ValueError("Generation request differs from the locally approved model")
        remaining=_timestamp(request['deadline'])-time.time()
        if not 0<remaining<=330: raise ValueError("Generation request deadline is invalid")
        return request

    @contextmanager
    def _lease(self, job):
        done=threading.Event();cancelled=threading.Event();lock=threading.Lock();state={"text":"","sequence":0};failures=[]
        def beat():
            with lock: body=dict(state);state['sequence']+=1
            response=self._action(job,"heartbeat",**body)
            if _timestamp(response.get('leaseExpiresAt'))<=time.time(): raise ValueError("Generation lease expired")
        def update(text):
            with lock: state['text']=text
        def renew():
            while not done.wait(self.heartbeat_seconds):
                try:
                    if self.stop.is_set(): raise PermissionError("Generation stopped")
                    beat()
                except (ConnectorUnavailable,ConnectorRejected,PermissionError,ValueError) as error:
                    failures.append(error);cancelled.set();return
        beat()
        thread=threading.Thread(target=renew,name='generation-lease',daemon=True);thread.start()
        try:
            yield update,cancelled
            if failures or self.stop.is_set(): raise PermissionError("Generation lease was lost or stopped")
        finally:
            done.set();thread.join(timeout=16)
            cancelled.set()

    def _deliver(self, key, pending):
        try:
            result=self._action(pending,'complete',**pending['completion'])
            if result.get('completed') is not True: raise ConnectorUnavailable("Generation completion was not acknowledged")
        except (ConnectorRejected,PermissionError):
            self.outbox.delivered(key)
            return {"state":"lease_lost"}
        self.outbox.delivered(key)
        return {"state":"completed","jobId":pending['jobId']}

    def run_once(self):
        # Retry identical saved completion first; never generate twice after an
        # ambiguous response or publish partial text as a finished answer.
        for key,pending in self.outbox.pending(limit=1): return self._deliver(key,pending)
        if not self.registered: self.register()
        result=self._post('/jobs/claim',{'workerId':self.worker_id,'modelCommitment':self.model_commitment})
        job=result.get('job')
        if job is None: return {'state':'idle'}
        request=self._validate_job(job)
        try:
            with self._lease(job) as (update,cancelled):
                completion=self.runtime.generate(request['messages'],max_output_tokens=request['maxOutputTokens'],
                    max_output_characters=request['maxOutputCharacters'],timeout_seconds=min(180,_timestamp(request['deadline'])-time.time()),
                    on_progress=update,cancelled=cancelled)
            pending={'jobId':job['jobId'],'leaseToken':job['leaseToken'],'completion':completion}
            key='generation-'+job['jobId'];self.outbox.enqueue(key,pending)
            return self._deliver(key,pending)
        except OllamaError as error:
            result=self._action(job,'fail',code=error.code)
            if result.get('failed') is not True: raise ConnectorUnavailable("Generation failure was not acknowledged")
            return {'state':'failed','code':error.code}
