from copy import deepcopy
from datetime import datetime,timezone
import json
import time

import httpx
import pytest

from rateloop_evaluator.generation_worker import GenerationWorker
from rateloop_evaluator.connector import ConnectorUnavailable
from rateloop_evaluator.learning import provision_key
from rateloop_evaluator.ollama import OllamaError
from rateloop_evaluator.protocol import commitment
from rateloop_evaluator.storage import RuntimeStore
from test_ollama import runtime_fixture


def worker_fixture(tmp_path, *, lost=False, fail=None):
    real,_,_=runtime_fixture();model=real.identity();calls=[];state={'lost':lost,'served':False,'generations':0}
    class Runtime:
        def identity(self):return model
        def close(self):pass
        def generate(self,messages,**kwargs):
            state['generations']+=1
            if fail:raise OllamaError(fail)
            kwargs['on_progress']('completed local output')
            return {'text':'completed local output','finishReason':'stop','inputTokens':20,'outputTokens':4}
    job={'jobId':'gen-job','leaseToken':'lease_'+'x'*32,'request':{'schemaVersion':'rateloop.generation-request.v1','jobId':'gen-job',
        'model':model,'modelCommitment':commitment(model,'rateloop.generation-model.v1'),'messages':[{'role':'user','content':'private prompt never persisted'}],
        'maxOutputTokens':2048,'maxOutputCharacters':16000,'deadline':datetime.fromtimestamp(time.time()+300,timezone.utc).isoformat().replace('+00:00','Z')}}
    def handle(req):
        assert req.headers['authorization']=='Bearer private-key'
        body=json.loads(req.content);calls.append((req.url.path,body))
        if req.url.path.endswith('/models'):return httpx.Response(200,json={'modelCommitment':job['request']['modelCommitment']})
        if req.url.path.endswith('/claim'):
            response=job if not state['served'] else None;state['served']=True
            return httpx.Response(200,json={'job':response})
        assert body['workerId']=='worker-test' and body['leaseToken']==job['leaseToken']
        if req.url.path.endswith('/heartbeat'):
            assert body['sequence']==0 and body['text']==''
            return httpx.Response(200,json={'leaseExpiresAt':datetime.fromtimestamp(time.time()+120,timezone.utc).isoformat().replace('+00:00','Z')})
        if req.url.path.endswith('/complete'):
            if state['lost']:state['lost']=False;raise httpx.ReadTimeout('sensitive raw upstream',request=req)
            return httpx.Response(200,json={'completed':True})
        if req.url.path.endswith('/fail'):return httpx.Response(200,json={'failed':True})
        pytest.fail(str(req.url))
    key=tmp_path/'key';provision_key(key);outbox=RuntimeStore(tmp_path/'generation.sqlite',key)
    worker=GenerationWorker(connection={'baseUrl':'https://rateloop.example','apiKey':'private-key'},worker_id='worker-test',model=model,
        runtime=Runtime(),outbox=outbox,transport=httpx.MockTransport(handle))
    return worker,calls,state,job,outbox


def test_worker_registers_pinned_identity_fences_completion_and_retains_no_input(tmp_path):
    worker,calls,state,job,outbox=worker_fixture(tmp_path)
    assert worker.run_once()=={'state':'completed','jobId':'gen-job'}
    assert worker.run_once()=={'state':'idle'}
    assert state['generations']==1 and not outbox.pending()
    assert b'private prompt never persisted' not in outbox.path.read_bytes()
    complete=[v for p,v in calls if p.endswith('/complete')]
    assert complete[0]['text']=='completed local output' and complete[0]['finishReason']=='stop'
    assert not any('messages' in v for _,v in calls)


def test_lost_completion_retries_identical_encrypted_result_without_second_inference(tmp_path):
    worker,calls,state,_,outbox=worker_fixture(tmp_path,lost=True)
    with pytest.raises(ConnectorUnavailable) as e:worker.run_once()
    assert 'sensitive raw upstream' not in str(e.value)
    assert len(outbox.pending())==1 and b'completed local output' not in outbox.path.read_bytes()
    assert worker.run_once()=={'state':'completed','jobId':'gen-job'}
    complete=[v for p,v in calls if p.endswith('/complete')]
    assert complete[0]==complete[1] and state['generations']==1 and not outbox.pending()


@pytest.mark.parametrize('code',['model_unavailable','context_overflow','generation_failed','output_limit'])
def test_runtime_failure_is_explicit_never_partial_completion(tmp_path,code):
    worker,calls,_,_,_=worker_fixture(tmp_path,fail=code)
    assert worker.run_once()=={'state':'failed','code':code}
    assert not any(p.endswith('/complete') for p,_ in calls)
    assert next(v for p,v in calls if p.endswith('/fail'))['code']==code


def test_job_cannot_choose_remote_url_or_another_model(tmp_path):
    worker,_,_,job,_=worker_fixture(tmp_path)
    for mutate in (lambda r:r.update(endpoint='https://attacker.example'),lambda r:r['model'].update(model='other')):
        changed=deepcopy(job);mutate(changed['request'])
        with pytest.raises(ValueError):worker._validate_job(changed)


def test_lost_authorization_cancels_local_inference_before_completion(tmp_path,monkeypatch):
    worker,calls,_,_,outbox=worker_fixture(tmp_path)
    original=worker._action
    beats=[0]
    def action(job,action,**body):
        if action=='heartbeat':
            beats[0]+=1
            if beats[0]>1:raise PermissionError('revoked')
        if action=='fail':raise PermissionError('revoked')
        return original(job,action,**body)
    worker._action=action
    seen=[]
    def generate(messages,**kwargs):
        assert kwargs['cancelled'].wait(2)
        seen.append('cancelled')
        raise OllamaError()
    worker.runtime.generate=generate
    with pytest.raises(PermissionError):worker.run_once()
    assert seen==['cancelled'] and not outbox.pending()
    assert not any(p.endswith('/complete') for p,_ in calls)
