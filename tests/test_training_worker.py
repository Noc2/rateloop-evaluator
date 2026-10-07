"""Authenticated website jobs exercise real import/registry/leases; only neural math is stubbed."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import time

import httpx
import pytest

from rateloop_evaluator import cli, training_worker
from rateloop_evaluator.backends import validate_local_model, write_model_manifest
from rateloop_evaluator.connector import RateLoopConnector, ConnectorUnavailable
from rateloop_evaluator.protocol import EvaluationRequest, commitment
from rateloop_evaluator.storage import RuntimeStore
from rateloop_evaluator.templates import custom_text_seed, custom_text_evaluation
from test_cli import initialized, invoke, model_files


def iso(value): return datetime.fromtimestamp(value,timezone.utc).isoformat(timespec="milliseconds").replace("+00:00","Z")


@pytest.fixture
def runner(initialized,tmp_path,capsys,monkeypatch):
    model=model_files(tmp_path); template=custom_text_seed("en")
    request=EvaluationRequest(workspaceId="workspace-test",caseId="seed",idempotencyKey="seed-case",modelBundleId="base",
        template=template,input={"text":"Synthetic"})
    request_path=tmp_path/"request.json"; cli.write_private(request_path,request.model_dump())
    invoke(capsys,initialized,"register","--activate","--custom-text","--model-dir",model,"--request",request_path)
    template=custom_text_evaluation("en","Does this document meet the required format?","Meets format","Needs changes")
    request.template=template
    root,local,store,registry=cli.state(SimpleNamespace(state_dir=initialized))
    rows=[{"caseId":f"case-{i}","sourceGroupId":f"source-{i}","input":{"text":f"Private example {i}","context":"","evidence":""},
        "labels":{"judgment":"approved" if i%2 else "rejected"}} for i in range(12)]
    digest=commitment({"workspaceId":"workspace-test","templateCommitment":request.template_commitment(),"provenance":"owner","rows":rows},
        "rateloop.evaluator.dataset.v1")
    authorization={"grantId":"dataset-permission","workspaceId":"workspace-test","apiKeyId":"api-1","workerId":"operator-1",
        "rights":["private_training"],"caseIds":[r["caseId"] for r in rows],"templateIds":[template.id],
        "templateCommitments":[request.template_commitment()],"fields":["input.text","input.context","input.evidence","imported_labels"],
        "modelBundleIds":["base"],"expiresAt":iso(time.time()+86400),"authorizationUntil":iso(time.time()+120)}
    job={"jobId":"training-job","action":"train","modelBundleId":"base","candidateBundleId":"candidate",
        "templateCommitment":request.template_commitment(),"datasetVersionId":"dataset-version","datasetCommitment":digest,
        "leaseToken":"training-job-fencing-token","leaseExpiresAt":iso(time.time()+120)}
    authorization.update(datasetVersionId=job["datasetVersionId"],datasetCommitment=job["datasetCommitment"])
    behavior={"pending":deepcopy(job),"claimed":False,"completed":[],"failed":[],"permission":True,"cancelled":False,"offline_complete":False,
        "train_calls":0,"cancel_during_train":False,"complete_status":200,"ambiguous_complete":False,"fail_status":200,"ambiguous_fail":False,"content_mutator":lambda value:None}
    calls=[]
    def transport(req):
        calls.append(req)
        assert req.headers["authorization"]=="Bearer scoped-operator-api-key"
        path=req.url.path; body=json.loads(req.content) if req.content else None
        if path.endswith("/training/workers/heartbeat"):
            authorization["authorizationUntil"]=iso(time.time()+120)
            return httpx.Response(200,json={"workerId":"operator-1","state":body["state"],"lastSeenAt":iso(time.time()),
                "permissions":[deepcopy(authorization)] if behavior["permission"] else []})
        if path.endswith("/claim"):
            if behavior["claimed"]: return httpx.Response(200,json={"job":None})
            behavior["claimed"]=True
            return httpx.Response(200,json={"job":behavior["pending"]})
        if path.endswith("/content"):
            assert req.headers["x-evaluator-lease"]==job["leaseToken"] and req.headers["x-evaluator-worker"]=="operator-1"
            content={key:value for key,value in behavior["pending"].items() if key!="leaseToken"}
            content.update(workspaceId="workspace-test",template=template.model_dump(),authorization=deepcopy(authorization))
            if content["action"] in ("train","compare"):
                content.update(dataset={"versionId":"dataset-version","provenance":"owner","rows":deepcopy(rows)},
                    recipe={"method":"lora","epochs":1,"maxSteps":20})
            behavior["content_mutator"](content)
            return httpx.Response(200,json=content)
        assert body["workerId"]=="operator-1" and body["leaseToken"]==job["leaseToken"]
        if path.endswith("/heartbeat"):
            return httpx.Response(410 if behavior["cancelled"] else 200,json={"leaseExpiresAt":iso(time.time()+120)})
        if path.endswith("/complete"):
            if behavior["offline_complete"]: raise httpx.ConnectError("Synthetic interruption",request=req)
            if behavior["complete_status"]!=200: return httpx.Response(behavior["complete_status"],json={"state":"cancelled"})
            behavior["completed"].append(body["result"])
            return httpx.Response(200,json={"status":"completed","jobId":"wrong-job" if behavior["ambiguous_complete"] else behavior["pending"]["jobId"]})
        if path.endswith("/fail"):
            assert set(body) == {"workerId", "leaseToken", "errorCode"}
            if behavior["fail_status"] != 200:
                return httpx.Response(behavior["fail_status"], json={"code": "synthetic_failure"})
            behavior["failed"].append(body)
            return httpx.Response(200,json={"status":"failed","jobId":"wrong-job" if behavior["ambiguous_fail"] else behavior["pending"]["jobId"]})
        raise AssertionError(path)
    connector=RateLoopConnector(base_url="https://rateloop.example",api_key="scoped-operator-api-key",api_key_id="api-1",
        workspace_id="workspace-test",agent_id="agent-1",agent_version_id="version-1",learning=store,
        runtime=RuntimeStore(root/"runtime.sqlite",local["encryptionKey"]),metadata_upload_enabled=True,transport=httpx.MockTransport(transport))
    class Backend:
        def __init__(self,path,device): self.manifest=validate_local_model(path)
        def count_tokens(self,*_): return 25
        def predict(self,_text,questions): return {q["id"]:{"approved":.8,"rejected":.2} for q in questions}
    def train(store,snapshot_id,workspace,source,output,*,bundle_id,options):
        behavior["train_calls"]+=1
        assert options.method=="lora" and options.epochs==behavior.get("expected_epochs",1) and options.max_steps==behavior.get("expected_steps",20)
        if behavior["cancel_during_train"]:
            behavior["cancelled"]=True
            time.sleep(1.1)
        snapshot=store.load_snapshot(snapshot_id,workspace)
        destination=Path(output)/"model"; destination.parent.mkdir(parents=True)
        shutil.copytree(source,destination)
        destination.joinpath("model.safetensors").write_bytes(b"Changed synthetic optimizer weights")
        source_manifest=validate_local_model(source)
        metadata={"bundleId":bundle_id,"workspaceId":workspace,"snapshotId":snapshot_id,"reloadMaxAbsoluteError":0.0,
            "trainingGroupIds":[r["group_id"] for r in snapshot["train"]],"trainingExampleIds":[r["evaluation_id"] for r in snapshot["train"]]}
        source_metadata={**source_manifest["source"],"baseWeightsSha256":source_manifest["files"]["model.safetensors"]}
        write_model_manifest(destination,source=source_metadata,training=metadata)
        store.register_model_lineage(bundle_id,snapshot_id,workspace)
        return {"modelDir":str(destination)}
    monkeypatch.setattr(training_worker,"GLiNERBackend",Backend)
    monkeypatch.setattr(training_worker,"train_snapshot",train)
    changed=[]
    worker=training_worker.TrainingWorker(connector,registry,state_dir=root,worker_id="operator-1",model_dir=model,
        model_bundle_ids=["base"],heartbeat_seconds=1,on_models_changed=lambda ids:changed.append(ids))
    return worker,behavior,authorization,job,template,store,registry,changed,calls


def next_job(behavior,job,action,target="candidate"):
    behavior["pending"]={**job,"jobId":action+"-job","action":action,"candidateBundleId":target,"leaseExpiresAt":iso(time.time()+120)}
    behavior["claimed"]=False


def test_independent_feedback_cycle_compares_frozen_incumbent_and_erases_withdrawn_examples(runner):
    worker,behavior,authorization,job,template,store,registry,changed,calls=runner
    rows=[]; now=time.time(); tc=job["templateCommitment"]
    for i in range(200):
        lineage={"aiExposed":False,"humanResultCommitment":f"human-{i}","auditIds":[f"audit-{i}"],
            "templateCommitment":tc,"inputCommitment":f"input-{i}","modelBundleId":"native-source-v2","trainingModelBundleId":"base"}
        rows.append({"caseId":f"human-{i}","sourceGroupId":f"source-{i}","input":{"text":f"Independent example {i}","context":"","evidence":""},
            "labels":{"judgment":"approved" if i%2 else "rejected"},"inputCommitment":f"input-{i}",
            "createdAt":iso(now-1000+i),"observedAt":iso(now-999+i),"lineage":lineage,
            "evidenceFingerprint":commitment(lineage,"rateloop.evaluator.feedback-evidence.v1"),
            "partition":"train" if i<140 else "calibration" if i<170 else "test"})
    digest=commitment({"workspaceId":"workspace-test","templateCommitment":tc,"provenance":"independent_human","rows":rows},"rateloop.evaluator.dataset.v1")
    authorization.update(caseIds=[row["caseId"] for row in rows],fields=["input.text","input.context","input.evidence","human_labels"],datasetCommitment=digest)
    job["datasetCommitment"]=digest; behavior["pending"]["datasetCommitment"]=digest
    behavior.update(expected_epochs=5,expected_steps=25)
    def mutate(content):
        content["dataset"]={"versionId":job["datasetVersionId"],"provenance":"independent_human","rows":deepcopy(rows)}
        content["recipe"]={"schemaVersion":training_worker.RECIPE_SCHEMA,"method":"lora","epochs":5,"maxSteps":25,"learningRate":.0001,
            "validationFraction":.2,"validationInterval":25,"earlyStoppingPatience":3,"minValidationPerLabel":5}
        content["comparisonModelBundleId"]="base"
    behavior["content_mutator"]=mutate
    assert worker.run_once()["state"]=="training_completed"
    result=behavior["completed"][0]
    assert result["comparison"]["independent_reference_count"]==30
    assert result["comparison"]["incumbent_model_bundle_id"]=="base"
    assert set(result["comparison"]["models"])=={"baseline","candidate","incumbent","reference"}
    assert result["signedManifest"]["manifest"]["synthetic"] is False
    assert not changed, "Training never activates a candidate"
    assert "Independent example" not in json.dumps(result)
    behavior["permission"]=False
    worker.sync_permissions()
    with store.transaction() as state:
        snapshot=state["snapshots"][result["snapshotId"]]
        assert snapshot["invalidated_at"] is not None
        assert all(not snapshot[part] for part in ("train","calibration","test"))
    with pytest.raises(PermissionError): registry.get("candidate","workspace-test")


def test_owner_dataset_train_compare_export_activate_and_rollback_on_same_runner(runner):
    worker,behavior,_,job,template,store,registry,changed,calls=runner
    assert worker.run_once()["state"]=="training_completed"
    result=behavior["completed"][0]
    assert result["comparison"]["independent_reference_count"]==0
    assert result["comparison"]["quality_gate"] is False
    assert set(result["comparison"]["models"])=={"baseline","candidate"}
    assert result["candidateRegistration"]["modelBundleId"]=="candidate"
    assert "taskCapability" not in result["candidateRegistration"]
    assert result["candidateRegistration"]["template"]==template.model_dump()
    assert result["signedManifest"]["manifest"]["id"]=="candidate"
    with pytest.raises(PermissionError,match="explicit activation"):
        registry.serving_policy("candidate","workspace-test",template)
    assert registry.serving_policy("base","workspace-test",template)["bundle_id"]=="base"
    with store.transaction() as db:
        assert not db["evaluations"] and not db["feedback"]
        assert all(g["rights"]==["private_training"] for g in db["grants"].values())
    assert "Private example" not in json.dumps(result)
    assert "scoped-operator-api-key" not in json.dumps(result)
    next_job(behavior,job,"activate")
    assert worker.run_once()["state"]=="training_completed"
    assert changed[-1]==["base","candidate"]
    assert behavior["completed"][-1]["candidateRegistration"]==result["candidateRegistration"]
    assert registry.serving_policy("candidate","workspace-test",template)["mode"]=="shadow"
    next_job(behavior,job,"rollback","base")
    assert worker.run_once()["state"]=="training_completed"
    assert registry.active("workspace-test",job["templateCommitment"],"en")["bundle_id"]=="base"
    assert behavior["train_calls"]==1
    next_job(behavior,job,"compare",None)
    assert worker.run_once()["state"]=="training_completed"
    assert behavior["completed"][-1]["snapshotId"]==result["snapshotId"]
    behavior["permission"]=False
    worker.sync_permissions()
    with pytest.raises(PermissionError): registry.get("candidate","workspace-test")
    assert worker.configured_bundles()==["base"]
    with store.transaction() as db:
        assert not db["dataset_examples"]
        assert all(not snapshot[part] for snapshot in db["snapshots"].values() for part in ("train","calibration","test"))


def test_retry_after_upload_failure_reuses_exact_completed_candidate_without_retraining(runner):
    worker,behavior,_,_,_,_,_,_,_=runner
    behavior["offline_complete"]=True
    with pytest.raises(ConnectorUnavailable): worker.run_once()
    saved=worker._saved()["result"]
    behavior["offline_complete"]=False
    assert worker.run_once()["state"]=="training_completed"
    assert behavior["completed"]==[saved] and behavior["train_calls"]==1


def test_cancelled_job_cannot_publish_checkpoint_or_comparison(runner):
    worker,behavior,_,_,_,store,registry,_,_=runner
    behavior["cancel_during_train"]=True
    assert worker.run_once()["state"]=="training_failed"
    assert behavior["train_calls"]==1 and not behavior["completed"]
    with pytest.raises(KeyError): registry.get("candidate","workspace-test")
    with store.transaction() as db: assert not db["lineage"]


@pytest.mark.parametrize("change",["workspace","template","dataset","recipe","candidate","authorization"])
def test_malformed_or_rebound_job_never_trains(runner,change):
    worker,behavior,_,_,_,store,_,_,_=runner
    def mutate(content):
        if change=="workspace": content["workspaceId"]="foreign"
        elif change=="template": content["template"]["questions"][0]["text"]="Changed task?"
        elif change=="dataset": content["dataset"]["rows"][0]["input"]["text"]="Changed material"
        elif change=="recipe": content["recipe"]["maxSteps"]=201
        elif change=="candidate": content["candidateBundleId"]="different-candidate"
        else: content["authorization"]["apiKeyId"]="other-key"
    behavior["content_mutator"]=mutate
    assert worker.run_once()["state"]=="training_failed"
    assert behavior["train_calls"]==0 and not behavior["completed"]


@pytest.mark.parametrize("change",["rights","apiKeyId","workerId","expiresAt","authorizationUntil","caseIds"])
def test_worker_permission_is_purpose_recipient_scope_and_time_bound(runner,change):
    worker,behavior,authorization,_,_,_,_,_,_=runner
    if change=="rights": authorization[change]=["ai_use","private_training"]
    elif change in ("apiKeyId","workerId"): authorization[change]="other"
    elif change=="expiresAt": authorization[change]=iso(time.time()+31*86400)
    elif change=="caseIds": authorization[change]=[]
    else:
        # The mock refreshes authorizationUntil itself, so inspect the validation boundary directly.
        authorization[change]=iso(time.time()+3600)
        with pytest.raises(PermissionError): worker._authorization(authorization)
        return
    with pytest.raises((ValueError,PermissionError)): worker.run_once()
    assert behavior["train_calls"]==0


def test_permission_cannot_silently_broaden_after_first_lease(runner):
    worker,_,authorization,_,_,store,_,_,_=runner
    worker.sync_permissions()
    authorization["caseIds"].append("another-case")
    with pytest.raises(PermissionError,match="new grant identity"): worker.sync_permissions()
    with store.transaction() as db: assert all(g["revoked_at"] is not None for g in db["grants"].values())


def test_rejected_activation_completion_leaves_previous_model_selected(runner):
    worker,behavior,_,job,template,_,registry,changed,_=runner
    assert worker.run_once()["state"]=="training_completed"
    next_job(behavior,job,"activate")
    behavior["complete_status"]=410
    assert worker.run_once()["state"]=="training_failed"
    assert not changed
    assert registry.serving_policy("base","workspace-test",template)["bundle_id"]=="base"
    with pytest.raises(PermissionError,match="explicit activation"):
        registry.serving_policy("candidate","workspace-test",template)


def test_acknowledged_switch_is_reconciled_before_any_more_inference(runner,monkeypatch):
    worker,behavior,_,job,template,_,registry,changed,_=runner
    assert worker.run_once()["state"]=="training_completed"
    next_job(behavior,job,"activate")
    apply=worker._apply_switch
    monkeypatch.setattr(worker,"_apply_switch",lambda *_:(_ for _ in ()).throw(RuntimeError("Synthetic restart before local finalize")))
    with pytest.raises(ConnectorUnavailable,match="reconciliation"): worker.run_once()
    assert worker._saved()["serverAcknowledged"] is True
    assert not changed
    with pytest.raises(PermissionError): registry.serving_policy("candidate","workspace-test",template)
    monkeypatch.setattr(worker,"_apply_switch",apply)
    assert worker.run_once()["state"]=="training_completed"
    assert changed==[["base","candidate"]]
    assert behavior["completed"][-1]==behavior["completed"][-2]
    assert behavior["train_calls"]==1
    assert registry.serving_policy("candidate","workspace-test",template)["mode"]=="shadow"


def test_permission_revoked_after_switch_acknowledgement_never_installs_candidate(runner,monkeypatch):
    worker,behavior,_,job,template,store,registry,changed,_=runner
    assert worker.run_once()["state"]=="training_completed"
    next_job(behavior,job,"activate")
    monkeypatch.setattr(worker,"_apply_switch",lambda *_:(_ for _ in ()).throw(RuntimeError("Interrupted finalize")))
    with pytest.raises(ConnectorUnavailable): worker.run_once()
    behavior["permission"]=False
    assert worker.run_once()["state"]=="training_permission_revoked"
    assert worker._saved() is None and changed==[["base"]]
    with pytest.raises(PermissionError): registry.get("candidate","workspace-test")
    assert registry.serving_policy("base","workspace-test",template)["bundle_id"]=="base"
    with store.transaction() as db: assert not db["dataset_examples"]


def test_retention_deadline_erases_rows_even_if_website_is_unavailable(runner,monkeypatch):
    worker,_,_,_,_,store,registry,_,_=runner
    assert worker.run_once()["state"]=="training_completed"
    with store.transaction() as db:
        for permission in worker._state(db)["permissions"].values():
            permission["authorization"]["expiresAt"]=iso(time.time()-1)
    monkeypatch.setattr(worker.connector,"_request",lambda *_args,**_kw:(_ for _ in ()).throw(ConnectorUnavailable("Offline")))
    with pytest.raises(ConnectorUnavailable): worker.sync_permissions()
    with store.transaction() as db: assert not db["dataset_examples"]
    with pytest.raises(PermissionError): registry.get("candidate","workspace-test")


def test_website_import_selects_exact_dataset_permission_among_overlapping_local_grants(runner):
    worker,_,authorization,_,_,store,registry,_,_=runner
    unrelated=store.add_grant(workspace_id="workspace-test",rights=["private_training"],
        case_ids=authorization["caseIds"],template_ids=authorization["templateIds"],
        template_commitments=authorization["templateCommitments"],model_bundle_ids=["base"],
        fields=authorization["fields"],expires_at=time.time()+3600,evidence="Separate owner-authorized dataset")
    assert worker.run_once()["state"]=="training_completed"
    with store.transaction() as db:
        selected=worker._state(db)["permissions"][authorization["grantId"]]["localId"]
        assert all(version["grant_ids"]==[selected] for version in db["datasets"].values())
        assert all(snapshot["grant_ids"]==[selected] for snapshot in db["snapshots"].values())
    store.revoke_grant(unrelated["id"],"workspace-test")
    assert registry.get("candidate","workspace-test")["manifest"]["id"]=="candidate"


def test_ambiguous_completion_never_applies_switch_and_retries_exact_result(runner):
    worker,behavior,_,job,template,_,registry,changed,_=runner
    assert worker.run_once()["state"]=="training_completed"
    next_job(behavior,job,"activate")
    behavior["ambiguous_complete"]=True
    with pytest.raises(ConnectorUnavailable,match="ambiguous"): worker.run_once()
    saved=worker._saved()["result"]
    assert not changed
    with pytest.raises(PermissionError): registry.serving_policy("candidate","workspace-test",template)
    behavior["ambiguous_complete"]=False
    assert worker.run_once()["state"]=="training_completed"
    assert behavior["completed"][-1]==saved and changed==[["base","candidate"]]


@pytest.mark.parametrize("failure", ["rejected", "unavailable", "ambiguous"])
def test_failure_report_is_durable_until_matching_acknowledgment(runner, failure):
    worker, behavior, _, _, _, _, _, _, calls = runner
    behavior["complete_status"] = 400
    behavior["fail_status"] = {"rejected": 400, "unavailable": 503, "ambiguous": 200}[failure]
    behavior["ambiguous_fail"] = failure == "ambiguous"
    with pytest.raises(ConnectorUnavailable):
        worker.run_once()
    saved = worker._saved()
    assert saved["failureCode"] == "local_training_validation_failed"
    assert "result" in saved
    completed_calls = len([r for r in calls if r.url.path.endswith("/complete")])
    behavior["fail_status"] = 200
    behavior["ambiguous_fail"] = False
    assert worker.run_once()["state"] == "training_failed"
    assert worker._saved() is None
    assert behavior["train_calls"] == 1
    assert len([r for r in calls if r.url.path.endswith("/complete")]) == completed_calls
    assert set(behavior["failed"][-1]) == {"workerId", "leaseToken", "errorCode"}


@pytest.mark.parametrize("status", [404, 409, 410])
def test_fenced_failure_intent_releases_local_job_without_retraining(runner, status):
    worker, behavior, _, _, _, _, _, _, _ = runner
    behavior["complete_status"] = 400
    behavior["fail_status"] = status
    assert worker.run_once()["state"] == "training_lease_lost"
    assert worker._saved() is None
    assert behavior["train_calls"] == 1


def test_reclaimed_failed_operation_retains_terminal_intent(runner):
    worker, behavior, _, _, _, _, _, _, calls = runner
    behavior["complete_status"] = 400
    behavior["fail_status"] = 409
    assert worker.run_once()["state"] == "training_lease_lost"
    completion_count = len([r for r in calls if r.url.path.endswith("/complete")])
    behavior["claimed"] = False
    behavior["complete_status"] = 200
    behavior["fail_status"] = 200
    assert worker.run_once()["state"] == "training_failed"
    assert worker._saved() is None
    assert behavior["train_calls"] == 1
    assert not behavior["completed"]
    assert len([r for r in calls if r.url.path.endswith("/complete")]) == completion_count


def test_reviewed_validation_recipe_cannot_expand_compute_or_change_selection():
    recipe={"schemaVersion":training_worker.RECIPE_SCHEMA,"method":"lora","epochs":5,"maxSteps":200,
        "validationFraction":.2,"validationInterval":25,"earlyStoppingPatience":3,"minValidationPerLabel":5,"learningRate":.0001}
    options=training_worker.training_options(recipe,"cpu")
    assert options.max_steps==200 and options.validation_fraction==.2 and options.min_validation_per_label==5
    assert training_worker.CAPABILITY["recipeSchemaVersion"]==recipe["schemaVersion"]
    for field,bad in (("maxSteps",201),("maxSteps",24),("maxSteps",True),("epochs",1),("validationFraction",0),
        ("learningRate",.1),("validationInterval",1),("minValidationPerLabel",1),("schemaVersion","other")):
        with pytest.raises(ValueError): training_worker.training_options({**recipe,field:bad},"cpu")
    with pytest.raises(ValueError): training_worker.training_options({**recipe,"path":"/arbitrary"},"cpu")


def test_insufficient_validation_groups_report_actionable_failure_without_private_details(runner,monkeypatch):
    worker,behavior,*_=runner
    def insufficient(*args,**kwargs): raise ValueError('Insufficient training data: private example must not leak')
    monkeypatch.setattr(training_worker,'train_snapshot',insufficient)
    assert worker.run_once()['state']=='training_failed'
    assert behavior['failed'][0]['errorCode']=='insufficient_training_data'
    assert 'private example' not in json.dumps(behavior['failed'])


def test_validation_regression_never_becomes_generic_failure_or_candidate(runner,monkeypatch):
    worker,behavior,*_=runner
    def declined(*args,**kwargs): raise ValueError('Validation did not improve over the original evaluator')
    monkeypatch.setattr(training_worker,'train_snapshot',declined)
    assert worker.run_once()['state']=='training_failed'
    assert behavior['failed'][0]['errorCode']=='validation_not_improved'
    assert not behavior['completed']


def test_interrupted_unsigned_checkpoint_retries_same_job_without_overwriting_artifact(runner):
    worker,behavior,_,job,template,store,registry,*_=runner
    incomplete=worker.root/'training-candidates'/job['jobId']/'checkpoints'
    incomplete.mkdir(parents=True)
    partial=incomplete/'partial-adapter.bin'
    partial.write_bytes(b'Private interrupted training checkpoint')
    assert worker.run_once()['state']=='training_completed'
    assert behavior['train_calls']==1
    preserved=list((worker.root/'training-interrupted'/job['jobId']).glob('attempt-*/checkpoints/partial-adapter.bin'))
    assert len(preserved)==1 and preserved[0].read_bytes()==b'Private interrupted training checkpoint'
    candidate=registry.get('candidate','workspace-test')
    assert candidate['artifact_root']==str(worker.root/'training-candidates'/job['jobId']/'model')
    assert candidate['manifest']['snapshot_id']==behavior['completed'][0]['snapshotId']


@pytest.mark.parametrize('reload_verified',[False,True])
def test_restart_distinguishes_initial_manifest_from_reload_verified_finalization(runner,monkeypatch,reload_verified):
    worker,behavior,_,job,template,store,registry,*_=runner
    train=training_worker.train_snapshot
    interrupted=[False]
    def partial(*args,**kwargs):
        report=train(*args,**kwargs)
        if not interrupted[0]:
            interrupted[0]=True
            # Recreate the exact boundaries before durable lineage exists.
            with store.transaction() as database: database['lineage'].pop('candidate')
            if not reload_verified:
                model=validate_local_model(report['modelDir'])
                model['training'].pop('reloadMaxAbsoluteError')
                write_model_manifest(report['modelDir'],source=model['source'],training=model['training'])
            raise ConnectorUnavailable('Synthetic process interruption before finalization')
        return report
    monkeypatch.setattr(training_worker,'train_snapshot',partial)
    with pytest.raises(ConnectorUnavailable): worker.run_once()
    assert worker.run_once()['state']=='training_completed'
    assert behavior['train_calls']==(1 if reload_verified else 2)
    assert registry.get('candidate','workspace-test')['manifest']['snapshot_id']==behavior['completed'][0]['snapshotId']
