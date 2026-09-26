"""One shared custom-task contract across registration, workers and inference."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

from fastapi import HTTPException
import pytest

from rateloop_evaluator import cli, hosted, worker_runtime
from rateloop_evaluator.protocol import EvaluationRequest, Template, commitment
from rateloop_evaluator.registry import BundleRegistry
from rateloop_evaluator.templates import (CUSTOM_TEXT_CAPABILITY, bundle_supports_template,
    custom_text_evaluation, custom_text_seed, is_custom_text_template, overall_approval, website_binary_question)
from test_cli import initialized, invoke, model_files
from test_hosted import config
from test_worker import website
from test_connector import setup


def register_custom(state, model_dir, tmp_path, capsys, bundle="base-custom", language="en"):
    request=EvaluationRequest.model_validate({"workspaceId":"workspace-test","caseId":"seed","idempotencyKey":"seed-key",
        "modelBundleId":bundle,"template":custom_text_seed(language).model_dump(),"input":{"text":"Synthetic seed"}})
    path=tmp_path/(bundle+".json"); path.write_text(request.model_dump_json())
    invoke(capsys,state,"register","--activate","--custom-text","--model-dir",model_dir,"--request",path)
    return request,path


@pytest.mark.parametrize("language",["en","de"])
def test_task_contract_is_shared_by_registration_worker_and_runtime(initialized,tmp_path,capsys,monkeypatch,language):
    model=model_files(tmp_path)
    seed,path=register_custom(initialized,model,tmp_path,capsys,language=language)
    root,local,store,registry=cli.state(SimpleNamespace(state_dir=initialized))
    manifest=registry.get(seed.modelBundleId,seed.workspaceId)["manifest"]
    request=seed.model_copy(deep=True)
    request.template=custom_text_evaluation(language,"Is this summary faithful to the source?","Faithful","Not faithful")
    assert request.template_commitment()!=seed.template_commitment()
    assert website_binary_question(request.template).id=="judgment"
    assert bundle_supports_template(manifest,request.template)
    assert registry.serving_policy(request.modelBundleId,request.workspaceId,request.template)["mode"]=="shadow"
    from rateloop_evaluator.storage import RuntimeStore
    connector=SimpleNamespace(workspace_id=request.workspaceId,learning=store,runtime=RuntimeStore(root/"runtime.sqlite",local["encryptionKey"]))
    connector._state=lambda db:db.setdefault("connector-test",{})
    seen=[]
    class Backend:
        def __init__(self,*_): pass
        def load(self): pass
        def count_tokens(self,*_): return 32
        def predict(self,text,questions):
            seen.append(questions)
            return {"judgment":{"approved":.9,"rejected":.1}}
    monkeypatch.setattr(worker_runtime,"GLiNERBackend",Backend)
    evaluate=worker_runtime.prepare_evaluator(connector,registry,[request.modelBundleId])
    # Bundle capability is not permission for any template; seed permission is insufficient.
    store.add_grant(workspace_id=request.workspaceId,rights=["ai_use"],expires_at=time.time()+60,
        evidence="owner seed permission",model_bundle_ids=[request.modelBundleId],template_commitments=[seed.template_commitment()])
    with pytest.raises(HTTPException) as failure: evaluate(request)
    assert failure.value.status_code==403 and len(seen)==1  # warmup only
    grant=store.add_grant(workspace_id=request.workspaceId,rights=["ai_use"],expires_at=time.time()+60,
        evidence="owner exact custom permission",model_bundle_ids=[request.modelBundleId],template_commitments=[request.template_commitment()])
    result=evaluate(request)
    assert result["outcome"]=="uncertain" and result["abstainReason"]=="uncalibrated"
    assert result["criteria"][0]["label"]=="approved" and result["criteria"][0]["probabilities"] is None
    assert seen[-1]==[q.model_dump() for q in request.template.questions]
    assert evaluate(request)==result and len(seen)==2
    with store.transaction() as state:
        assert state["evaluations"][request.input_commitment()]["input"] is None
    changed=request.model_copy(deep=True); changed.template.questions[0].text="Another question?"
    with pytest.raises(HTTPException) as failure: evaluate(changed)
    assert failure.value.status_code==403
    foreign=request.model_copy(deep=True); foreign.workspaceId="other-workspace"
    with pytest.raises(HTTPException) as failure: evaluate(foreign)
    assert failure.value.status_code==403
    store.revoke_grant(grant["id"],request.workspaceId)
    with pytest.raises(HTTPException) as failure: evaluate(request)
    assert failure.value.status_code==403
    # Exporting a derived task remains metadata only and carries no calibration claim.
    path.write_text(request.model_dump_json())
    output=tmp_path/"registration.json"
    invoke(capsys,initialized,"export-registration","--bundle-id",request.modelBundleId,"--request",path,"--output",output)
    registration=json.loads(output.read_text())
    assert registration["taskCapability"]==CUSTOM_TEXT_CAPABILITY
    assert registration["template"]==request.template.model_dump()
    assert registration["templateCommitment"]==request.template_commitment()
    assert registration["criteria"][0]["calibrationId"] is None


@pytest.mark.parametrize("mutate",[
    lambda t:t.update(id="other"),lambda t:t.update(version=2),lambda t:t.update(maxTokens=513),
    lambda t:t["questions"][0].update(id="other"),lambda t:t["questions"][0].update(text=" padded "),
    lambda t:t["questions"][0]["labels"][0].update(description=" "*3),
    lambda t:t["questions"][0]["labels"][0].update(description="x"*41),
    lambda t:t["questions"][0]["labels"][0].update(description="NO"),
    lambda t:t["questions"][0].update(text="x"*501),
    lambda t:t["questions"][0].update(text="Line\nbreak"),
    lambda t:t["questions"][0].update(text="\U0001f600"*251),
    lambda t:t["questions"][0].update(passLabels=["rejected"]),
    lambda t:t["questions"][0]["labels"].reverse(),lambda t:t["questions"].append({**t["questions"][0],"id":"second"}),
])
def test_all_custom_consumers_reject_noncanonical_boundary_cases(mutate):
    template=custom_text_seed("en").model_dump(); mutate(template)
    template=Template.model_validate(template)
    bundle={"languages":["en"],"max_tokens":512,"template_commitments":[],"task_capability":CUSTOM_TEXT_CAPABILITY}
    assert not is_custom_text_template(template)
    assert not bundle_supports_template(bundle,template)
    with pytest.raises(ValueError): website_binary_question(template)


@pytest.mark.parametrize("positive,negative,allowed",[("YES","yes",False),("ß","SS",True),("İ","I",True),(" Yes ","No",True)])
def test_portable_label_normalization(positive,negative,allowed):
    if not allowed:
        with pytest.raises(ValueError): custom_text_evaluation("en","Question?",positive,negative)
    else:
        template=custom_text_evaluation("en"," Question? ",positive,negative)
        assert is_custom_text_template(template)
        assert template.questions[0].text=="Question?"


def test_hosted_addition_keeps_legacy_identities_and_reuses_artifact_paths(config):
    original=hosted.bootstrap(config)
    old={p:Path(p).read_bytes() for p in original["registrations"]}
    upgraded=deepcopy(config)
    upgraded["bundles"] += [{"language":language,"modelBundleId":"custom-"+language,"taskCapability":CUSTOM_TEXT_CAPABILITY}
                             for language in ("en","de")]
    path=Path(config["stateDir"])/"hosted.json"; cli.write_private(path,upgraded)
    assert hosted.read_config(path)==upgraded
    first=hosted.bootstrap(upgraded)
    assert len(first["registrations"])==4
    assert all(Path(p).read_bytes()==data for p,data in old.items())
    assert hosted.bootstrap(upgraded)==first
    _,_,store,registry=cli.state(SimpleNamespace(state_dir=config["stateDir"]))
    with store.transaction() as db: assert not db["grants"]
    for language in ("en","de"):
        record=registry.get("custom-"+language,config["workspaceId"])
        assert record["artifact_root"]==config["modelDir"]
    with pytest.raises(PermissionError): hosted.bootstrap(config)  # cannot discard queued bundle identities


def test_plain_and_trained_bundles_do_not_gain_custom_scope():
    template=custom_text_seed("en")
    base={"languages":["en"],"template_commitments":["sha256:"+"0"*64],"max_tokens":512}
    assert not bundle_supports_template(base,template)
    for extra in ({"snapshot_id":"private-trained"},{"calibrations":[{"id":"other-rubric"}]}):
        assert not bundle_supports_template({**base,"task_capability":CUSTOM_TEXT_CAPABILITY,**extra},template)


def test_worker_processes_custom_question_and_syncs_its_exact_label_mapping(website,monkeypatch):
    worker,req,backend,remote,behavior,calls=website
    req.template=custom_text_evaluation("en","Does the document include the required sections?","Complete","Incomplete")
    behavior["job"].update(templateCommitment=req.template_commitment(),inputCommitment=req.input_commitment())
    behavior["content"]["request"]=req.model_dump()
    for consent in remote["consents"]: consent["templateCommitments"]=[req.template_commitment()]
    from rateloop_evaluator.service import Principal, create_app
    principal=Principal(req.workspaceId,frozenset({"evaluate"}))
    monkeypatch.setattr(backend,"predict",lambda *_:{"judgment":{"approved":.8,"rejected":.2}})
    app=create_app(backend=backend,bundle={"id":req.modelBundleId,"languages":["en"],"template_commitments":[],
        "task_capability":CUSTOM_TEXT_CAPABILITY},learning=worker.connector.learning,runtime=worker.connector.runtime,
        tokens={"0"*64:principal})
    worker.evaluate=lambda r:app.state.evaluate(r,principal)
    assert worker.run_once()["state"]=="completed"
    imported=[]
    monkeypatch.setattr(worker.connector,"fetch_and_import_labels",lambda consent_id,**kwargs:
        imported.append(kwargs) or {"imported":1,"rejected":[],"truncated":False})
    assert worker.sync_labels()["imported"]==1
    assert imported==[{"question_id":"judgment","template_commitment":req.template_commitment(),
                      "outcome_labels":{"positive":"approved","negative":"rejected"}}]


def test_portable_fixtures_bind_full_question_and_labels():
    fixtures=json.loads((Path(__file__).parent/"fixtures/custom-text-templates.json").read_text())
    for item in fixtures:
        template=Template.model_validate(item["template"])
        assert is_custom_text_template(template)
        assert commitment(template.model_dump(),"rateloop.evaluator.template.v1")==item["templateCommitment"]
        altered=template.model_copy(deep=True); altered.questions[0].text+="!"
        assert commitment(altered.model_dump(),"rateloop.evaluator.template.v1")!=item["templateCommitment"]


@pytest.mark.parametrize("change",["capability","revision","scope","limit","calibration","snapshot"])
def test_custom_capability_registration_rejects_unsupported_model_or_scope(initialized,tmp_path,capsys,change):
    model=model_files(tmp_path)
    seed,_=register_custom(initialized,model,tmp_path,capsys)
    _,_,_,registry=cli.state(SimpleNamespace(state_dir=initialized))
    manifest=registry.get(seed.modelBundleId,seed.workspaceId)["manifest"]
    manifest["id"]="invalid-custom"
    if change=="capability": manifest["task_capability"]={"schemaVersion":"unknown"}
    elif change=="revision": manifest["model_revision"]="0"*40
    elif change=="scope": manifest["template_commitments"]=["sha256:"+"0"*64]
    elif change=="limit": manifest["max_tokens"]=8192
    elif change=="calibration": manifest["calibrations"]=[{}]
    else: manifest["snapshot_id"]="private-training"
    with pytest.raises(ValueError,match="Custom task capability"):
        registry.register(manifest,seed.workspaceId,model)
