from argparse import Namespace
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import time

from fastapi.testclient import TestClient
import jsonschema
import pytest

from rateloop_evaluator import cli
from rateloop_evaluator.protocol import EvaluationRequest,EvaluationResult,commitment
from rateloop_evaluator.evidence import validate_evidence_binding,evidence_for_result,runtime_identity
from rateloop_evaluator.ollama_judge import OLLAMA_LABEL_ONLY_CAPABILITY,OllamaJudge,register_judge,adapter_commitment
from rateloop_evaluator.service import Principal,create_app
from rateloop_evaluator.storage import RuntimeStore
from test_ollama import runtime_fixture

ROOT=Path(__file__).resolve().parents[1]
FIXTURE=json.loads((ROOT/'tests/fixtures/label-only-v2.json').read_text())


def test_label_only_wire_and_evidence_parity_keep_all_confidence_explicitly_absent():
    request=EvaluationRequest.model_validate(FIXTURE['request']);result=EvaluationResult.model_validate(FIXTURE['result'])
    jsonschema.validate(FIXTURE['result'],EvaluationResult.model_json_schema())
    assert validate_evidence_binding(FIXTURE['evidence'],result,request).checks[0].judgment=='meets'
    source="""import assert from 'node:assert/strict';
import {parseEvaluationResult} from './contracts/evaluator.ts';
import {validateEvidenceBinding} from './contracts/evidence.ts';
const v=JSON.parse(process.argv[1]);
assert.deepEqual(parseEvaluationResult(v.result),v.result);
assert.deepEqual(validateEvidenceBinding(v.evidence,v.result,v.request),v.evidence);"""
    subprocess.run(['node','--experimental-strip-types','--input-type=module','-e',source,json.dumps(FIXTURE)],cwd=ROOT,check=True,capture_output=True,text=True)
    assert result.criteria[0].rawScores is None and result.outcome=='uncertain'


@pytest.mark.parametrize('change', [lambda r:r.update(schemaVersion='rateloop.evaluator.result.v1'),
    lambda r:r.update(outcome='pass'),lambda r:r.update(abstainReason=None),
    lambda r:r['criteria'][0].update(rawScores={'approved':1,'rejected':0}),
    lambda r:r['criteria'][0].update(probabilities={'approved':1,'rejected':0},calibrationId='fake')])
def test_label_only_result_rejects_version_downgrade_synthetic_scores_and_calibration(change):
    result=deepcopy(FIXTURE['result']);change(result)
    result['resultCommitment']=commitment({k:v for k,v in result.items() if k!='resultCommitment'},result['schemaVersion'])
    with pytest.raises(ValueError):EvaluationResult.model_validate(result)
    with pytest.raises(jsonschema.ValidationError):jsonschema.validate(result,EvaluationResult.model_json_schema())


def test_real_judge_adapter_consumes_only_valid_structured_labels(tmp_path):
    runtime,_,_=runtime_fixture(stream=[{'model':'qwen3.5:4b','message':{'content':'{"labels":{"overall_approval":"approved"}}'},
        'done':True,'done_reason':'stop','prompt_eval_count':80,'eval_count':12}])
    model=runtime.identity()
    cli.write_private(tmp_path/'ollama-judge.json',{'schemaVersion':'rateloop.ollama-judge.v1','baseUrl':'http://127.0.0.1:11434','model':model,'adapterCommitment':adapter_commitment(model)})
    judge=OllamaJudge(tmp_path);judge.runtime.close();judge.runtime=runtime
    req=EvaluationRequest.model_validate(FIXTURE['request'])
    assert judge.predict(req.input.render(),[q.model_dump() for q in req.template.questions])=={'overall_approval':'approved'}
    assert judge.count_tokens(req.input.render(),[q.model_dump() for q in req.template.questions])<512
    judge.unload()


def test_judge_registered_runtime_service_retains_no_training_input_even_with_separate_grant(tmp_path):
    root=tmp_path/'state';cli.run(Namespace(command='init',state_dir=str(root),workspace='workspace-label-only'))
    _,config,store,registry=cli.state(Namespace(state_dir=str(root)))
    runtime,_,_=runtime_fixture();model=runtime.identity()
    bundles,exports=register_judge(root=root,registry=registry,store=store,workspace=config['workspaceId'],worker_id='worker-test',model=model,base_url='http://127.0.0.1:11434')
    assert len(bundles)==2 and all(e['scoreCapability']==OLLAMA_LABEL_ONLY_CAPABILITY for e in exports)
    assert all(e['adapterCommitment']==adapter_commitment(model) for e in exports)
    request=EvaluationRequest.model_validate({**FIXTURE['request'],'modelBundleId':bundles[0]['modelBundleId']})
    with pytest.raises(PermissionError,match='shadow mode only'):
        registry.promote(request.modelBundleId,config['workspaceId'],template_commitment=request.template_commitment(),language='en',mode='assisted')
    store.add_grant(workspace_id=config['workspaceId'],rights=['ai_use','private_training'],expires_at=time.time()+3600,evidence='synthetic test')
    record=registry.get(request.modelBundleId,config['workspaceId'])
    judge=OllamaJudge(record['artifact_root']);judge.runtime.close()
    class Runtime:
        def generate(self,*args,**kwargs):return {'text':'{"labels":{"overall_approval":"approved"}}'}
    judge.runtime=Runtime()
    identity=Principal(config['workspaceId'],frozenset({'evaluate'}))
    app=create_app(backend=judge,bundle=record['manifest'],learning=store,runtime=RuntimeStore(root/'runtime.sqlite',config['encryptionKey']),tokens={'0'*64:identity},
        validate_bundle=lambda r:registry.serving_policy(r.modelBundleId,config['workspaceId'],r.template))
    result=app.state.evaluate_with_evidence(request,identity)
    assert result['result']['schemaVersion']=='rateloop.evaluator.result.v2'
    assert result['result']['criteria'][0]['rawScores'] is None
    assert result['evidence']['identity']['scoreAdapter']==OLLAMA_LABEL_ONLY_CAPABILITY['adapter']
    assert result['evidence']['checks'][0]['judgment']=='meets'
    with store.transaction() as db:assert all(row['input'] is None for row in db['evaluations'].values())


def test_judge_freezes_prompt_adapter_and_complete_runtime_identity(tmp_path,monkeypatch):
    from rateloop_evaluator import ollama_judge
    runtime,_,_=runtime_fixture();model=runtime.identity();runtime.close()
    descriptor={'schemaVersion':'rateloop.ollama-judge.v1','baseUrl':'http://127.0.0.1:11434','model':model,'adapterCommitment':adapter_commitment(model)}
    cli.write_private(tmp_path/'ollama-judge.json',descriptor)
    for field,value in [('templateDigest','sha256:'+'c'*64),('runtimeVersion','0.35.2')]:
        changed={**descriptor,'model':{**model,field:value}}
        cli.write_private(tmp_path/'ollama-judge.json',changed)
        with pytest.raises(ValueError,match='adapter changed'):OllamaJudge(tmp_path)
    cli.write_private(tmp_path/'ollama-judge.json',descriptor)
    monkeypatch.setattr(ollama_judge,'_SYSTEM','Changed adapter prompt')
    with pytest.raises(ValueError,match='adapter changed'):OllamaJudge(tmp_path)


def test_judge_preserves_full_prompt_overflow_as_distinct_non_prediction(tmp_path, monkeypatch):
    from rateloop_evaluator.ollama import OllamaError
    from rateloop_evaluator.ollama_judge import JudgeInputOverflow
    runtime, _, _ = runtime_fixture(); model = runtime.identity()
    cli.write_private(tmp_path/'ollama-judge.json', {'schemaVersion': 'rateloop.ollama-judge.v1',
        'baseUrl': 'http://127.0.0.1:11434', 'model': model, 'adapterCommitment': adapter_commitment(model)})
    judge = OllamaJudge(tmp_path); judge.runtime.close(); judge.runtime = runtime
    def overflow(*args, **kwargs): raise OllamaError('context_overflow')
    monkeypatch.setattr(runtime, 'generate', overflow)
    request = EvaluationRequest.model_validate(FIXTURE['request'])
    with pytest.raises(JudgeInputOverflow):
        judge.predict(request.input.render(), [q.model_dump() for q in request.template.questions])
    judge.unload()
