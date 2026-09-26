"""Explicit operator-owned training runner. No hosted opt-in or weight uploads.

Website jobs select bounded built-in recipes and immutable data, never executable
code or arbitrary paths. Local dataset grants and job leases are separate: a
cancelled job stops promptly without falsely revoking a still-authorized dataset.
"""
from __future__ import annotations

from argparse import Namespace
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import threading
import time

from . import cli
from .backends import GLiNERBackend, MODEL_ID, MODEL_REVISION, MANIFEST_NAME, validate_local_model
from .comparison import compare_snapshot
from .connector import ConnectorRejected, ConnectorUnavailable, _hash, _opaque, _timestamp
from .datasets import import_dataset, erase_dataset_version, MAX_BYTES, MAX_ROWS
from .execution import model_execution, ExecutionBusy
from .protocol import EvaluationRequest, Template, commitment
from .templates import is_custom_text_template, overall_approval
from .training import TrainOptions, train_snapshot

CAPABILITY={"schemaVersion":"rateloop.evaluator.training-worker.v1","maxRows":MAX_ROWS,"maxSteps":200}
_JOB_FIELDS={"jobId","action","modelBundleId","candidateBundleId","templateCommitment","datasetVersionId",
             "datasetCommitment","leaseToken","leaseExpiresAt"}
_AUTH_FIELDS={"grantId","workspaceId","apiKeyId","workerId","rights","caseIds","templateIds","templateCommitments",
              "fields","modelBundleIds","expiresAt","authorizationUntil","datasetVersionId","datasetCommitment"}


class _JobStore:
    """Every optimizer step/snapshot read checks cancellation and current lease."""
    def __init__(self, store, check): self.store,self.check=store,check
    def __getattr__(self,name): return getattr(self.store,name)
    def load_snapshot(self,*args,**kwargs):
        self.check()
        return self.store.load_snapshot(*args,**kwargs)


class TrainingWorker:
    def __init__(self, connector, registry, *, state_dir, worker_id, model_dir, model_bundle_ids, device="cpu",
                 heartbeat_seconds=15, on_models_changed=None):
        _opaque(worker_id)
        if device not in ("cpu","mps","cuda") or not 1<=heartbeat_seconds<=30:
            raise ValueError("Invalid training runner configuration")
        model=validate_local_model(model_dir)
        if model.get("training") or (model["source"].get("repository"),model["source"].get("revision"))!=(MODEL_ID,MODEL_REVISION):
            raise ValueError("Training runner requires the explicitly provisioned pinned public base")
        self.connector,self.registry=connector,registry
        self.root=Path(state_dir).resolve(); self.model_dir=str(Path(model_dir).resolve())
        self.worker_id,self.device,self.heartbeat_seconds=worker_id,device,heartbeat_seconds
        self.base_bundle_ids=list(model_bundle_ids)
        self.on_models_changed=on_models_changed or (lambda _:None)

    def _state(self, database):
        return self.connector._state(database).setdefault("training_worker",{}).setdefault(self.worker_id,
            {"permissions":{},"job":None,"active_bundles":[],"operations":{}})

    def configured_bundles(self):
        with self.connector.learning.transaction() as db:
            candidates=list(self._state(db)["active_bundles"])
        allowed=[]
        for bundle_id in candidates:
            try:
                try: self.registry.get(bundle_id,self.connector.workspace_id,verify_artifacts=False)
                except ValueError: self.registry.get(bundle_id,self.connector.workspace_id)
            except (KeyError,PermissionError): continue
            allowed.append(bundle_id)
        return list(dict.fromkeys([*self.base_bundle_ids,*allowed]))

    def _authorization(self, value):
        if not isinstance(value,dict) or set(value)!=_AUTH_FIELDS:
            raise ValueError("Training authorization fields do not match the contract")
        if (value["workspaceId"],value["apiKeyId"],value["workerId"]) != (
                self.connector.workspace_id,self.connector.api_key_id,self.worker_id):
            raise PermissionError("Training authorization recipient mismatch")
        _opaque(value["grantId"]); _opaque(value["datasetVersionId"]); _hash(value["datasetCommitment"])
        if value["rights"]!=["private_training"]:
            raise PermissionError("Training permission cannot grant inference, sharing or publication")
        limits={"caseIds":MAX_ROWS,"templateIds":1,"templateCommitments":1,"modelBundleIds":1,"fields":4}
        for key,limit in limits.items():
            values=value[key]
            if not isinstance(values,list) or not 1<=len(values)<=limit or any(not isinstance(item,str) for item in values) or len(set(values))!=len(values):
                raise ValueError("Training authorization scope must be explicit and unique")
            for item in values:
                (_hash if key=="templateCommitments" else _opaque)(item)
        if "imported_labels" not in value["fields"] or not set(value["fields"])<= {"input.text","input.context","input.evidence","imported_labels"}:
            raise ValueError("Training authorization must cover imported labels and supported fields")
        now=time.time(); expires=_timestamp(value["expiresAt"]); until=_timestamp(value["authorizationUntil"])
        if not now<until<=min(expires,now+150) or not now<expires<=now+30*86400+30:
            raise PermissionError("Training permission retention or execution lease is invalid")
        return {key:item for key,item in value.items() if key!="authorizationUntil"},expires,until

    def sync_permissions(self, state="ready"):
        with self.connector.learning.transaction() as db:
            previous=deepcopy(self._state(db)["permissions"])
        # Enforce the durable retention deadline even when the website is down.
        for old in previous.values():
            if _timestamp(old["authorization"]["expiresAt"])<=time.time(): self._erase_permission(old)
        body={"workerId":self.worker_id,"modelBundleIds":self.configured_bundles(),"capability":dict(CAPABILITY),"state":state}
        response=self.connector._request("POST","/training/workers/heartbeat",json=body)
        if response.get("workerId")!=self.worker_id or response.get("state") not in ("ready","training"):
            raise ValueError("Training heartbeat changed worker identity")
        _timestamp(response.get("lastSeenAt"))
        supplied=response.get("permissions")
        if not isinstance(supplied,list) or len(supplied)>50:
            raise ValueError("Training heartbeat requires bounded explicit dataset permissions")
        validated={}
        for authorization in supplied:
            immutable,expires,until=self._authorization(authorization)
            identity=authorization["grantId"]
            if identity in validated: raise ValueError("Duplicate dataset permission")
            validated[identity]=(authorization,immutable,expires,until)
        for grant_id,old in previous.items():
            if grant_id not in validated:
                self._erase_permission(old)
        mirrored={}
        for grant_id,(authorization,immutable,expires,until) in validated.items():
            digest=commitment(immutable,"rateloop.dataset-permission.v1")
            local_id="dataset_"+hashlib.sha256((self.connector.namespace+self.worker_id+grant_id).encode()).hexdigest()[:48]
            old=previous.get(grant_id)
            if old:
                if old["digest"]!=digest:
                    self._erase_permission(old)
                    raise PermissionError("Dataset permission changed without a new grant identity")
                self.connector.learning.renew_authorization(local_id,self.connector.workspace_id,until)
            else:
                self.connector.learning.add_grant(workspace_id=self.connector.workspace_id,rights=["private_training"],
                    case_ids=authorization["caseIds"],template_ids=authorization["templateIds"],fields=authorization["fields"],
                    model_bundle_ids=authorization["modelBundleIds"],template_commitments=authorization["templateCommitments"],
                    expires_at=expires,authorization_until=until,evidence="Owner-authorized website dataset "+grant_id,grant_id=local_id)
            mirrored[grant_id]={"localId":local_id,"digest":digest,"authorization":deepcopy(authorization)}
        with self.connector.learning.transaction() as db: self._state(db)["permissions"]=mirrored
        return response

    def _erase_permission(self, permission):
        self.connector.learning.revoke_grant(permission["localId"],self.connector.workspace_id)
        with self.connector.learning.transaction() as db:
            versions=[key for key,value in db.get("datasets",{}).items()
                if value["workspace_id"]==self.connector.workspace_id and permission["localId"] in value["grant_ids"]]
        for version_id in versions:
            erase_dataset_version(self.connector.learning,version_id,self.connector.workspace_id)

    def _has_dataset_permission(self, job):
        with self.connector.learning.transaction() as db:
            return any(value["authorization"]["datasetVersionId"]==job["datasetVersionId"] and
                value["authorization"]["datasetCommitment"]==job["datasetCommitment"]
                for value in self._state(db)["permissions"].values())

    def _saved(self):
        with self.connector.learning.transaction() as db: return deepcopy(self._state(db)["job"])

    def _save(self, job):
        with self.connector.learning.transaction() as db:
            state=self._state(db)
            state["job"]=deepcopy(job)
            if job is not None:
                operations=state.setdefault("operations",{})
                operations[job["jobId"]]=deepcopy(job)
                while len(operations)>50: operations.pop(next(iter(operations)))

    def _restore_progress(self,job):
        with self.connector.learning.transaction() as db:
            old=deepcopy(self._state(db).setdefault("operations",{}).get(job["jobId"]))
        if old:
            if any(old[key]!=job[key] for key in _JOB_FIELDS-{"leaseToken","leaseExpiresAt"}):
                raise ValueError("Reclaimed training job changed immutable identity")
            job.update({key:value for key,value in old.items() if key not in _JOB_FIELDS})
        return job

    def _post(self, job, action, **extra):
        return self.connector._request("POST",f"/training/jobs/{job['jobId']}/{action}",
            json={"workerId":self.worker_id,"leaseToken":job["leaseToken"],**extra})

    def _validate_claim(self, job):
        if not isinstance(job,dict) or set(job)!=_JOB_FIELDS or job["action"] not in ("compare","train","activate","rollback"):
            raise ValueError("Invalid training job contract")
        for key in ("jobId","modelBundleId","datasetVersionId"): _opaque(job[key])
        _hash(job["templateCommitment"]); _hash(job["datasetCommitment"])
        if job["modelBundleId"] not in self.base_bundle_ids:
            raise PermissionError("Training job selected an unconfigured public base")
        if job["candidateBundleId"] is not None: _opaque(job["candidateBundleId"])
        if job["action"] in ("train","activate","rollback") and job["candidateBundleId"] is None:
            raise ValueError("Candidate identity is required")
        if not isinstance(job["leaseToken"],str) or not 16<=len(job["leaseToken"])<=512 or any(c in job["leaseToken"] for c in "\r\n"):
            raise ValueError("Invalid training fencing token")
        if not time.time()<_timestamp(job["leaseExpiresAt"])<=time.time()+150:
            raise PermissionError("Training lease expired or exceeds allowed duration")

    def _heartbeat(self,job):
        response=self._post(job,"heartbeat")
        until=_timestamp(response.get("leaseExpiresAt"))
        if not time.time()<until<=time.time()+150: raise PermissionError("Invalid renewed training lease")
        job["leaseExpiresAt"]=response["leaseExpiresAt"]
        self._save(job)

    @contextmanager
    def _renew(self,job):
        stopped=threading.Event(); failures=[]
        def check():
            if failures: raise failures[0]
            if _timestamp(job["leaseExpiresAt"])<=time.time(): raise PermissionError("Training job lease expired")
        def renew():
            while not stopped.wait(self.heartbeat_seconds):
                try:
                    self._heartbeat(job)
                    self.sync_permissions("training")
                except Exception as error:
                    failures.append(error); return
        thread=threading.Thread(target=renew,name="training-job-lease",daemon=True); thread.start()
        try: yield check
        finally:
            stopped.set(); thread.join(timeout=35)
            if thread.is_alive(): failures.append(ConnectorUnavailable("Training lease refresh did not finish"))

    def _content(self,job):
        content=self.connector._request("GET",f"/training/jobs/{job['jobId']}/content",
            headers={"X-Evaluator-Lease":job["leaseToken"],"X-Evaluator-Worker":self.worker_id})
        if content.get("workspaceId")!=self.connector.workspace_id:
            raise PermissionError("Training content workspace mismatch")
        for key in _JOB_FIELDS-{"leaseToken","leaseExpiresAt"}:
            if content.get(key)!=job[key]: raise ValueError("Training content changed committed job identity")
        template=Template.model_validate(content.get("template"))
        if not is_custom_text_template(template) and template!=overall_approval(template.language):
            raise ValueError("Unsupported training question")
        if commitment(template.model_dump(),"rateloop.evaluator.template.v1")!=job["templateCommitment"]:
            raise ValueError("Training template commitment mismatch")
        authorization=content.get("authorization")
        immutable,_,_=self._authorization(authorization)
        with self.connector.learning.transaction() as db:
            local=self._state(db)["permissions"].get(authorization["grantId"])
        if not local or local["digest"]!=commitment(immutable,"rateloop.dataset-permission.v1"):
            raise PermissionError("Job permission is not in the current dataset authorization lease")
        if (job["templateCommitment"] not in authorization["templateCommitments"] or job["modelBundleId"] not in authorization["modelBundleIds"]
                or job["datasetVersionId"]!=authorization["datasetVersionId"] or job["datasetCommitment"]!=authorization["datasetCommitment"]):
            raise PermissionError("Training authorization does not match task and base model")
        return content,template

    def _snapshot(self,job,content,template):
        dataset=content.get("dataset")
        if not isinstance(dataset,dict) or set(dataset)!={"versionId","provenance","rows"} or dataset["versionId"]!=job["datasetVersionId"]:
            raise ValueError("Training dataset identity mismatch")
        rows=dataset["rows"]
        if not isinstance(rows,list) or not 1<=len(rows)<=MAX_ROWS: raise ValueError("Training row count exceeds limit")
        committed={"workspaceId":self.connector.workspace_id,"templateCommitment":job["templateCommitment"],
            "provenance":dataset["provenance"],"rows":rows}
        if commitment(committed,"rateloop.evaluator.dataset.v1")!=job["datasetCommitment"]:
            raise ValueError("Training dataset commitment mismatch")
        with self.connector.learning.transaction() as db:
            saved=deepcopy(self._state(db).setdefault("snapshots",{}).get(job["datasetVersionId"]))
        if saved:
            if saved["datasetCommitment"]!=job["datasetCommitment"]: raise ValueError("Dataset version changed bytes")
            return self.connector.learning.load_snapshot(saved["snapshotId"],self.connector.workspace_id)
        imported_rows=[]
        for row in rows:
            if not isinstance(row,dict) or set(row)!={"caseId","sourceGroupId","input","labels"}:
                raise ValueError("Training dataset row shape mismatch")
            if not isinstance(row["input"],dict) or set(row["input"])!={"text","context","evidence"}:
                raise ValueError("Training input fields must be explicit")
            if not isinstance(row["labels"],dict) or set(row["labels"])!={template.questions[0].id}:
                raise ValueError("Training labels do not match the exact question")
            imported_rows.append({"case_id":row["caseId"],"group_id":row["sourceGroupId"],**row["input"],"label":row["labels"][template.questions[0].id]})
        encoded="\n".join(json.dumps(row,ensure_ascii=False) for row in imported_rows).encode()
        if len(encoded)>MAX_BYTES: raise ValueError("Training dataset exceeds private import size limit")
        version=import_dataset(self.connector.learning,workspace_id=self.connector.workspace_id,dataset_id=job["datasetVersionId"],
            template=template,content=encoded,format="jsonl",provenance=dataset["provenance"],
            evidence="Explicit website training permission "+content["authorization"]["grantId"],model_bundle_id=job["modelBundleId"])
        snapshot=self.connector.learning.create_snapshot(self.connector.workspace_id,template.id,template.version,
            template_commitment=job["templateCommitment"],dataset_version_ids=[version["id"]],include_feedback=False)
        with self.connector.learning.transaction() as db:
            self._state(db).setdefault("snapshots",{})[job["datasetVersionId"]]={"datasetCommitment":job["datasetCommitment"],"snapshotId":snapshot["id"]}
        return snapshot

    def _registration(self,bundle_id,template):
        request=EvaluationRequest(workspaceId=self.connector.workspace_id,caseId="training-registration",idempotencyKey="training-registration",
            modelBundleId=bundle_id,template=template,input={"text":"Synthetic metadata export"})
        directory=self.root/"training-exports"/bundle_id
        request_file=directory/"request.json"; output=directory/"registration.json"
        cli.write_private(request_file,request.model_dump())
        cli.run(Namespace(command="export-registration",state_dir=str(self.root),bundle_id=bundle_id,request=str(request_file),output=str(output)))
        record=self.registry.get(bundle_id,self.connector.workspace_id)
        return json.loads(output.read_text()),record["envelope"],request_file

    def _operate(self,job,content,template,check):
        result={"schemaVersion":"rateloop.evaluator.training-result.v1","action":job["action"],"datasetVersionId":job["datasetVersionId"]}
        if job["action"] in ("activate","rollback"):
            try: active=self.registry.active(self.connector.workspace_id,job["templateCommitment"],template.language)
            except KeyError:
                if job["action"]!="activate": raise
                active={"bundle_id":job["modelBundleId"]}
            previous=active["bundle_id"]
            check()
            if job["action"]=="activate":
                candidate=self.registry.get(job["candidateBundleId"],self.connector.workspace_id)
                with self.connector.learning.transaction() as db:
                    reviewed=deepcopy(self._state(db).setdefault("candidates",{}).get(job["candidateBundleId"]))
                if (not reviewed or reviewed["datasetVersionId"]!=job["datasetVersionId"] or reviewed["datasetCommitment"]!=job["datasetCommitment"]):
                    raise PermissionError("Candidate is not bound to this runner's reviewed dataset")
                if candidate["manifest"].get("task_capability") or candidate["manifest"]["template_commitments"]!=[job["templateCommitment"]]:
                    raise PermissionError("Activation requires the exact trained task candidate")
                result.update(candidateRegistration=reviewed["registration"],signedManifest=reviewed["signedManifest"])
            else:
                self.registry.rollback(self.connector.workspace_id,job["templateCommitment"],template.language,
                    expected_bundle_id=job["candidateBundleId"],dry_run=True)
            if previous==job["candidateBundleId"]:
                raise ValueError("Requested model is already active")
            result.update(activeModelBundleId=job["candidateBundleId"],previousModelBundleId=previous)
            check()
            return result
        snapshot=self._snapshot(job,content,template)
        checked_store=_JobStore(self.connector.learning,check)
        result["snapshotId"]=snapshot["id"]
        models={"baseline":GLiNERBackend(self.model_dir,self.device)}
        candidate_id=job["candidateBundleId"]
        if job["action"]=="train":
            recipe=content.get("recipe")
            if (not isinstance(recipe,dict) or set(recipe)!={"method","epochs","maxSteps"} or recipe["method"]!="lora"
                    or type(recipe["epochs"]) is not int or recipe["epochs"]!=1 or type(recipe["maxSteps"]) is not int or not 1<=recipe["maxSteps"]<=CAPABILITY["maxSteps"]):
                raise ValueError("Only the bounded private LoRA recipe is supported")
            directory=self.root/"training-candidates"/job["jobId"]
            artifact=directory/"model"
            if artifact.joinpath(MANIFEST_NAME).is_file():
                model=validate_local_model(artifact)
                if (model.get("training",{}).get("bundleId"),model.get("training",{}).get("snapshotId"))!=(candidate_id,snapshot["id"]):
                    raise ValueError("Saved training artifact identity mismatch")
            else:
                report=train_snapshot(checked_store,snapshot["id"],self.connector.workspace_id,self.model_dir,directory,
                    bundle_id=candidate_id,options=TrainOptions(method="lora",device=self.device,epochs=1,max_steps=recipe["maxSteps"]))
                artifact=Path(report["modelDir"])
            check()
            try: self.registry.get(candidate_id,self.connector.workspace_id)
            except KeyError:
                request=EvaluationRequest(workspaceId=self.connector.workspace_id,caseId="training-registration",idempotencyKey="training-registration",
                    modelBundleId=candidate_id,template=template,input={"text":"Synthetic metadata export"})
                request_file=directory/"registration-request.json"; cli.write_private(request_file,request.model_dump())
                cli.run(Namespace(command="register",state_dir=str(self.root),model_dir=str(artifact),request=str(request_file),
                    snapshot_id=snapshot["id"],calibrations=None,real_data=False,selective_policy=None,custom_text=False,activate=False))
            candidate=self.registry.get(candidate_id,self.connector.workspace_id)
            if candidate["manifest"].get("snapshot_id")!=snapshot["id"] or candidate["artifact_root"]!=str(artifact.resolve()):
                raise ValueError("Registered candidate differs from this exact training artifact")
            registration,envelope,_=self._registration(candidate_id,template)
            with self.connector.learning.transaction() as db:
                candidates=self._state(db).setdefault("candidates",{})
                reviewed={"datasetVersionId":job["datasetVersionId"],"datasetCommitment":job["datasetCommitment"],
                    "registration":registration,"signedManifest":envelope}
                if candidate_id in candidates and candidates[candidate_id]!=reviewed: raise ValueError("Candidate review metadata changed")
                candidates[candidate_id]=reviewed
            result.update(candidateRegistration=registration,signedManifest=envelope)
        if candidate_id:
            candidate=self.registry.get(candidate_id,self.connector.workspace_id)
            if candidate["manifest"]["template_commitments"]!=[job["templateCommitment"]]:
                raise ValueError("Comparison candidate belongs to another exact task")
            models["candidate"]=GLiNERBackend(candidate["artifact_root"],self.device)
        result["comparison"]=compare_snapshot(checked_store,snapshot["id"],self.connector.workspace_id,models)
        check()
        return result

    def _apply_switch(self,job):
        """Finalize only the acknowledged intent, before the next inference poll."""
        if job["action"] not in ("activate","rollback"): return
        template=Template.model_validate(job["taskTemplate"])
        result=job["result"]
        try: active=self.registry.active(self.connector.workspace_id,job["templateCommitment"],template.language)
        except KeyError:
            if job["action"]!="activate": raise
            active=self.registry.promote(job["modelBundleId"],self.connector.workspace_id,
                template_commitment=job["templateCommitment"],language=template.language,mode="shadow",template=template)
        if active["bundle_id"]!=result["activeModelBundleId"]:
            if active["bundle_id"]!=result["previousModelBundleId"]:
                raise PermissionError("Local default changed outside the acknowledged switch")
            if job["action"]=="activate":
                self.registry.promote(job["candidateBundleId"],self.connector.workspace_id,
                    template_commitment=job["templateCommitment"],language=template.language,mode="shadow")
            else:
                self.registry.rollback(self.connector.workspace_id,job["templateCommitment"],template.language,
                    expected_bundle_id=job["candidateBundleId"])
        with self.connector.learning.transaction() as db:
            # Pending task evaluations were drained by the server before the
            # acknowledged switch. Preserve artifacts, not all warm checkpoints.
            active=self._state(db)["active_bundles"]
            active[:]=[bundle for bundle in active if job["templateCommitment"] not in
                db["bundles"].get(bundle,{}).get("envelope",{}).get("manifest",{}).get("template_commitments",[])]
            if job["candidateBundleId"] not in self.base_bundle_ids and job["candidateBundleId"] not in active:
                active.append(job["candidateBundleId"])
        self.on_models_changed(self.configured_bundles())

    def _complete(self,job):
        # The server acknowledges identical completed receipts idempotently even
        # after the original execution lease. Persist before and after that edge
        # so a crash cannot silently leave a different local model selected.
        response=self._post(job,"complete",result=job["result"])
        if response.get("jobId")!=job["jobId"] or response.get("status")!="completed":
            # An ambiguous HTTP success is not proof of the intended switch.
            # Keep the immutable result so the next attempt can reconcile it.
            raise ConnectorUnavailable("Training completion acknowledgment is ambiguous")
        job["serverAcknowledged"]=True; self._save(job)
        self.sync_permissions("training")
        self._apply_switch(job)
        self._save(None)
        return {"state":"training_completed","jobId":job["jobId"],"action":job["action"]}

    def run_once(self):
        self.sync_permissions()
        job=self._saved()
        if job and job.get("serverAcknowledged") and not self._has_dataset_permission(job):
            # A fresh server response proved the source was withdrawn after
            # acknowledgement. Retirement denies use; do not install its switch.
            self._save(None)
            self.on_models_changed(self.configured_bundles())
            return {"state":"training_permission_revoked","jobId":job["jobId"]}
        if job and "result" not in job:
            try: self._heartbeat(job)
            except ConnectorRejected as error:
                if error.status not in (404,409,410): raise
                self._save(None); return {"state":"training_lease_lost"}
        elif not job:
            job=self.connector._request("POST","/training/jobs/claim",json={"workerId":self.worker_id,
                "modelBundleIds":self.base_bundle_ids,"capability":dict(CAPABILITY)}).get("job")
            if job is None: return {"state":"idle"}
            self._validate_claim(job); job=self._restore_progress(job); self._save(job)
        try:
            with model_execution(self.connector.learning):
                self.sync_permissions("training")
                if "result" in job: return self._complete(job)
                content,template=self._content(job)
                job["taskTemplate"]=template.model_dump(); self._save(job)
                with self._renew(job) as check:
                    job["result"]=self._operate(job,content,template,check)
                    if len(json.dumps(job["result"]).encode())>256*1024: raise ValueError("Training metadata report exceeds limit")
                    check(); self._save(job)
                self._heartbeat(job)
                return self._complete(job)
        except ConnectorUnavailable:
            raise
        except ExecutionBusy:
            return {"state":"busy","reason":"local_model_operation"}
        except (ConnectorRejected,PermissionError,ValueError,RuntimeError,KeyError):
            if job.get("serverAcknowledged"):
                # Keep the intent for reconciliation; never start inference
                # using the old default while the server has selected another.
                raise ConnectorUnavailable("Acknowledged model switch requires local reconciliation") from None
            try: self._post(job,"fail",errorCode="local_training_validation_failed",retryable=False)
            except (ConnectorUnavailable,ConnectorRejected,PermissionError): pass
            self._save(None)
            return {"state":"training_failed","jobId":job["jobId"]}
        finally:
            try: self.sync_permissions("ready")
            except (ConnectorUnavailable,ConnectorRejected,PermissionError,ValueError): pass
