"""Exercise explicit connected rollback through the real CLI and signed registry."""
from argparse import Namespace
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from rateloop_evaluator import cli
from rateloop_evaluator.execution import ExecutionBusy, model_execution
from rateloop_evaluator.protocol import EvaluationRequest
from test_cli import model_files


@pytest.fixture
def operator_state(tmp_path, monkeypatch):
    spec=spec_from_file_location("alpha_operator_rollback_test",Path(__file__).parents[1]/"scripts/alpha_e2e_operator.py")
    operator=module_from_spec(spec);spec.loader.exec_module(operator)
    state=tmp_path/"state"; model=model_files(tmp_path)
    config=tmp_path/"operator.json"
    cli.write_private(config,{"stateDir":str(state),"workspaceId":"workspace-test","modelDir":str(model),
        "device":"cpu","baseBundlePrefix":"alpha-test","connector":{}})
    operator.run(Namespace(config=str(config),command="bootstrap"))
    _,_,store,registry=cli.state(Namespace(state_dir=str(state)))
    request=json.loads((tmp_path/"state-operator/en-request.json").read_text())
    template=EvaluationRequest.model_validate(request).template_commitment()
    request["modelBundleId"]="candidate"
    request_file=tmp_path/"candidate-request.json";cli.write_private(request_file,request)
    operator.call(state,"register","--model-dir",model,"--request",request_file)
    events=[]
    def sync():
        events.append("sync")
        return {"mode":"shadow"}
    connector=SimpleNamespace(learning=store,sync_grants=sync,close=lambda:events.append("close"))
    monkeypatch.setattr(operator,"connect",lambda *_:connector)
    args=Namespace(config=str(config),command="rollback",language="en",bundle_id=["alpha-test-en"])
    return operator,args,state,store,registry,template,connector,events


def test_operator_cli_registry_restore_only_the_explicit_previous_bundle(operator_state):
    operator,args,_,_,registry,template,_,events=operator_state
    assert registry.active("workspace-test",template,"en")["bundle_id"]=="candidate"
    assert operator.run(args)=={"modelBundleId":"alpha-test-en","activeModelBundleId":"alpha-test-en",
        "templateCommitment":template,"language":"en","mode":"shadow"}
    assert events==["sync","close"]
    assert registry.active("workspace-test",template,"en")["bundle_id"]=="alpha-test-en"
    # A second rollback cannot turn into an arbitrary activation.
    with pytest.raises(RuntimeError,match="Operator command failed: rollback"):
        operator.run(args)
    assert registry.active("workspace-test",template,"en")["bundle_id"]=="alpha-test-en"


@pytest.mark.parametrize("expected",["not-registered","alpha-test-de","candidate"])
def test_operator_expected_target_mismatch_preserves_active_candidate(operator_state,expected):
    operator,args,_,_,registry,template,_,events=operator_state
    args.bundle_id=[expected]
    with pytest.raises(RuntimeError,match="Operator command failed: rollback"):
        operator.run(args)
    assert registry.active("workspace-test",template,"en")["bundle_id"]=="candidate"
    assert events==["sync","close"]


@pytest.mark.parametrize("refusal",["off","revoked"])
def test_operator_checks_fresh_authorization_before_local_mutation(operator_state,refusal):
    operator,args,_,_,registry,template,connector,events=operator_state
    def sync():
        events.append("sync")
        if refusal=="revoked": raise PermissionError("Credential is revoked")
        return {"mode":"off"}
    connector.sync_grants=sync
    with pytest.raises(PermissionError): operator.run(args)
    assert registry.active("workspace-test",template,"en")["bundle_id"]=="candidate"
    assert events==["sync","close"]


def test_operator_and_registry_cannot_rollback_during_another_model_operation(operator_state):
    operator,args,_,store,registry,template,_,events=operator_state
    failures=[]
    def contend():
        for action in (lambda:operator.run(args),lambda:registry.rollback("workspace-test",template,"en",
                      expected_bundle_id="alpha-test-en")):
            try: action()
            except ExecutionBusy: failures.append("busy")
    with model_execution(store):
        thread=threading.Thread(target=contend);thread.start();thread.join(timeout=2)
        assert not thread.is_alive()
    assert failures==["busy","busy"] and events==["close"]
    assert registry.active("workspace-test",template,"en")["bundle_id"]=="candidate"


def test_operator_rejects_retired_previous_lineage_before_activation(operator_state):
    operator,args,_,store,registry,template,_,_=operator_state
    source=registry.get("alpha-test-en","workspace-test")
    manifest=source["manifest"]
    rubric=json.loads((Path(args.config).parent/"state-operator/en-request.json").read_text())["template"]
    grant=store.add_grant(workspace_id="workspace-test",rights=["ai_use","private_training"],
        expires_at=time.time()+3600,evidence="Synthetic local rollback regression")
    for index in range(3):
        identity=f"example-{index}"
        store.record_evaluation(evaluation_id=identity,workspace_id="workspace-test",case_id=identity,
            input_commitment=identity,template_commitment=template,template=rubric,input_payload={"text":identity})
        store.add_feedback(workspace_id="workspace-test",evaluation_id=identity,input_commitment=identity,
            template_commitment=template,annotator_id="synthetic",labels={"overall_approval":"approved"},
            exposed_to_ai=False,independent_human=True)
    snapshot=store.create_snapshot("workspace-test",rubric["id"],rubric["version"])
    store.register_model_lineage("learned-base",snapshot["id"],"workspace-test")
    manifest.update(id="learned-base",snapshot_id=snapshot["id"])
    registry.register(manifest,"workspace-test",source["artifact_root"])
    registry.promote("learned-base","workspace-test",template_commitment=template,language="en")
    registry.promote("candidate","workspace-test",template_commitment=template,language="en")
    store.revoke_grant(grant["id"],"workspace-test")
    args.bundle_id=["learned-base"]
    with pytest.raises(RuntimeError,match="Operator command failed: rollback"):
        operator.run(args)
    assert registry.active("workspace-test",template,"en")["bundle_id"]=="candidate"
