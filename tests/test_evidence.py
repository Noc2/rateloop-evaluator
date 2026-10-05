import copy
import json
from pathlib import Path
import subprocess

import pytest
from pydantic import ValidationError

from rateloop_evaluator.evidence import EvaluationEvidence, make_evidence, validate_evidence_binding
from rateloop_evaluator.protocol import EvaluationRequest, commitment, make_result

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    request = EvaluationRequest.model_validate_json((ROOT/'examples/reply-request.json').read_text())
    q=request.template.questions[0]
    result=make_result(workspaceId=request.workspaceId,caseId=request.caseId,modelBundleId=request.modelBundleId,
        inputCommitment=request.input_commitment(),templateCommitment=request.template_commitment(),outcome='uncertain',
        abstainReason='uncalibrated',criteria=[dict(questionId=q.id,label=q.labels[0].id,
            rawScores={label.id:(.8 if index == 0 else .2) for index,label in enumerate(q.labels)},probabilities=None,calibrationId=None)],
        durationMs=1,observedAt='2026-10-05T12:00:00.000Z')
    fields={key:getattr(result,key) for key in ('workspaceId','caseId','modelBundleId','inputCommitment','templateCommitment','resultCommitment','observedAt')}
    evidence=make_evidence(**fields,answerCommitment=commitment(request.input.text,'rateloop.evaluator.answer.v1'),
        language=request.template.language,identity={'runtime':'rateloop-evaluator/0.1.0','scoreAdapter':'synthetic-test-only'},
        checks=[dict(id='criterion:'+q.id,questionId=q.id,kind='criterion',state='completed',judgment='meets',
            coverage={'status':'complete','checkedUnits':1,'totalUnits':1},calibration={'status':'not_validated'})])
    return request,result,evidence


def ts_parse(value, result, request, valid):
    source='''
import assert from 'node:assert/strict';
import {parseEvaluationEvidence,validateEvidenceBinding} from './contracts/evidence.ts';
const [v,r,q,valid]=JSON.parse(process.argv[1]);
if(valid) assert.deepEqual(validateEvidenceBinding(v,r,q),v);
else assert.throws(()=>validateEvidenceBinding(v,r,q));
'''
    subprocess.run(['node','--experimental-strip-types','--input-type=module','-e',source,json.dumps([value,result.model_dump(),request.model_dump(),valid])],cwd=ROOT,check=True,capture_output=True,text=True)


def test_python_typescript_evidence_parity_and_v1_commitment_unchanged():
    request,result,evidence=fixture()
    assert validate_evidence_binding(evidence,result,request)==evidence
    ts_parse(evidence.model_dump(),result,request,True)
    assert result.resultCommitment==commitment(result.model_dump(exclude={'resultCommitment'}),'rateloop.evaluator.result.v1')


@pytest.mark.parametrize('change',[
    lambda v:v.update(caseId='another-case'),
    lambda v:v.update(answerCommitment='sha256:'+'a'*64),
    lambda v:v['checks'][0].update(judgment='does_not_meet'),
    lambda v:v['checks'][0].update(state='not_checked'),
    lambda v:v['checks'][0]['coverage'].update(status='partial'),
    lambda v:v['checks'][0]['calibration'].update(status='mapping_registered',calibrationId='fake-calibration'),
    lambda v:v['checks'].append(copy.deepcopy(v['checks'][0])),
])
def test_cross_language_invariants_and_exact_binding(change):
    request,result,evidence=fixture(); value=evidence.model_dump(); change(value)
    value['evidenceCommitment']=commitment({k:v for k,v in value.items() if k!='evidenceCommitment'},'rateloop.evaluator.evidence.v2')
    with pytest.raises(ValueError): validate_evidence_binding(value,result,request)
    ts_parse(value,result,request,False)


def test_unperformed_fact_check_has_no_judgment_or_source_claim():
    request,result,evidence=fixture(); value=evidence.model_dump()
    value['checks'].append(dict(id='factual_claims',questionId=None,kind='factual_claims',state='not_checked',judgment=None,
        reasonCode='external_sources_not_checked',coverage={'status':'none','checkedUnits':0,'totalUnits':None},sourceRefs=[],
        calibration={'status':'not_validated','calibrationId':None}))
    value['evidenceCommitment']=commitment({k:v for k,v in value.items() if k!='evidenceCommitment'},'rateloop.evaluator.evidence.v2')
    assert validate_evidence_binding(value,result,request).checks[-1].judgment is None
    ts_parse(value,result,request,True)
    assert request.input.text not in json.dumps(value)


@pytest.mark.parametrize('change',[
    lambda v:v['checks'].clear(),
    lambda v:v['checks'].append({**copy.deepcopy(v['checks'][0]),'id':'different-id'}),
    lambda v:v['checks'][0].update(kind='supplied_material',sourceRefs=[{'sourceId':'supplied-evidence','sourceCommitment':'sha256:'+'a'*64,'passageCommitment':'sha256:'+'a'*64,'startByte':0,'endByte':1}]),
    lambda v:v['checks'][0].update(kind='factual_claims'),
    lambda v:v['checks'][0].update(kind='code_execution'),
])
def test_no_hidden_criteria_or_fabricated_execution(change):
    request,result,evidence=fixture();value=evidence.model_dump();change(value)
    value['evidenceCommitment']=commitment({k:v for k,v in value.items() if k!='evidenceCommitment'},'rateloop.evaluator.evidence.v2')
    with pytest.raises(ValueError):validate_evidence_binding(value,result,request)
    ts_parse(value,result,request,False)


def test_supplied_material_unicode_hash_parity_and_semantic_label_binding():
    from rateloop_evaluator.source_evidence import supplied_material_template
    from rateloop_evaluator.evidence import RuntimeIdentity,evidence_for_result
    request,_,_=fixture();request.template=supplied_material_template('de');request.input.evidence='Grüße 😀: der Termin ist Donnerstag.'
    result=make_result(workspaceId=request.workspaceId,caseId=request.caseId,modelBundleId=request.modelBundleId,
        inputCommitment=request.input_commitment(),templateCommitment=request.template_commitment(),outcome='uncertain',abstainReason='uncalibrated',
        criteria=[{'questionId':'source_support','label':'contradicted','rawScores':{'supported':.1,'contradicted':.8,'insufficient_evidence':.1},'probabilities':None,'calibrationId':None}],durationMs=1,observedAt='2026-10-05T12:00:00.000Z')
    evidence=evidence_for_result(request,result,identity=RuntimeIdentity(runtime='fixture-only',scoreAdapter='fixture-only'))
    ts_parse(evidence.model_dump(),result,request,True)
    for mutate in (lambda v:v['checks'][0].update(judgment='meets'),lambda v:v['checks'][0]['sourceRefs'][0].update(endByte=3)):
        value=evidence.model_dump();mutate(value)
        value['evidenceCommitment']=commitment({k:v for k,v in value.items() if k!='evidenceCommitment'},'rateloop.evaluator.evidence.v2')
        with pytest.raises(ValueError):validate_evidence_binding(value,result,request)
        ts_parse(value,result,request,False)


def test_typescript_v2_client_binds_evidence_and_preserves_v1():
    request,result,evidence=fixture()
    source='''
import assert from 'node:assert/strict';
import {EvaluatorClient} from './clients/typescript/index.ts';
const [q,result,evidence]=JSON.parse(process.argv[1]);
const client=new EvaluatorClient({endpoint:'http://127.0.0.1:8765',token:'fixture',fetch:async url=>new Response(JSON.stringify(url.pathname==='/v1/evaluate'?result:{result,evidence}))});
assert.deepEqual(await client.evaluate(q),result);
assert.deepEqual(await client.evaluateWithEvidence(q),{result,evidence});
await assert.rejects(client.evaluateWithEvidence({...q,input:{...q.input,text:'changed'}}),/different case/);
'''
    subprocess.run(['node','--experimental-strip-types','--input-type=module','-e',source,json.dumps([request.model_dump(),result.model_dump(),evidence.model_dump()])],cwd=ROOT,check=True,capture_output=True,text=True)
