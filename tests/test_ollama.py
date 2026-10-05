from copy import deepcopy
import json
import threading

import httpx
import pytest

from rateloop_evaluator.ollama import OllamaRuntime, OllamaError
from rateloop_evaluator.protocol import commitment

MODEL='qwen3.5:4b'
SHOW={'modelfile':'FROM /private/models/blobs/sha256-'+ 'a'*64+'\n', 'template':'{{ .Prompt }}','parameters':'temperature 1','system':'',
      'model_info':{'qwen35.context_length':262144},'capabilities':['completion','thinking'],'details':{'quantization_level':'Q4_K_M'}}


def runtime_fixture(*, change=None, stream=None, rendered='Synthetic complete rendered prompt', render_supported=True):
    calls=[]; state={'version':'0.35.1','digest':'b'*64,'show':deepcopy(SHOW)}
    if change: change(state)
    def handle(request):
        body=json.loads(request.content) if request.content else None
        calls.append((request.url.path,body))
        assert request.url.host=='127.0.0.1'
        if request.url.path=='/api/version': return httpx.Response(200,json={'version':state['version']})
        if request.url.path=='/api/tags': return httpx.Response(200,json={'models':[{'name':MODEL,'digest':state['digest']}]})
        if request.url.path=='/api/show': return httpx.Response(200,json=state['show'])
        if request.url.path=='/api/chat':
            assert body['think'] is False and body['truncate'] is False and body['shift'] is False
            assert 'tools' not in body and body['options']['num_ctx']==8192
            assert 1<=body['options']['num_predict']<=2048
            if body.get('_debug_render_only'):
                return httpx.Response(200,json={'model':MODEL,'_debug_info':{'rendered_template':rendered}} if render_supported else {'message':{'content':'ignored render option'}})
            values=stream or [{'model':MODEL,'message':{'content':'Local '},'done':False},
                              {'model':MODEL,'message':{'content':'works'},'done':True,'done_reason':'stop','prompt_eval_count':12,'eval_count':2}]
            return httpx.Response(200,content=b''.join((json.dumps(v)+'\n').encode() for v in values))
        pytest.fail('Unexpected outbound request')
    return OllamaRuntime(model=MODEL,transport=httpx.MockTransport(handle)),calls,state


def test_pins_actual_weight_blob_complete_template_defaults_and_runtime_then_generates():
    runtime,calls,_=runtime_fixture()
    identity=runtime.identity()
    assert identity['weightDigest']=='sha256:'+'a'*64
    assert identity['templateDigest']==commitment({'manifestDigest':'sha256:'+'b'*64,'template':'{{ .Prompt }}','parameters':'temperature 1','system':'','messages':[]},'rateloop.ollama.template.v1')
    runtime.expected_identity=identity
    progress=[]
    result=runtime.generate([{'role':'user','content':'Synthetic private case'}],on_progress=progress.append)
    assert result=={'text':'Local works','finishReason':'stop','inputTokens':12,'outputTokens':2}
    assert progress[-1]=='Local works'
    assert all(path in ('/api/version','/api/tags','/api/show','/api/chat') for path,_ in calls)
    assert sum(path=='/api/chat' for path,_ in calls)==2
    runtime.close()


@pytest.mark.parametrize('mutate', [lambda s:s.update(version='0.34.9'),lambda s:s['show'].update(remote_host='https://cloud.example'),
    lambda s:s['show'].update(remote_model='cloud-model'),lambda s:s['show'].update(modelfile='FROM other-tag'),
    lambda s:s['show'].update(capabilities=['embedding']),lambda s:s.update(digest='invalid')])
def test_unverified_or_cloud_model_identity_is_rejected_before_content(mutate):
    runtime,calls,_=runtime_fixture(change=mutate)
    with pytest.raises(OllamaError):runtime.generate([{'role':'user','content':'private'}])
    assert not any(p=='/api/chat' for p,_ in calls)


def test_changed_local_digest_cannot_follow_old_approval():
    runtime,calls,state=runtime_fixture();runtime.expected_identity=runtime.identity();state['digest']='c'*64
    with pytest.raises(OllamaError,match='model_unavailable'):runtime.generate([{'role':'user','content':'private'}])
    assert not any(p=='/api/chat' for p,_ in calls)


@pytest.mark.parametrize('url',['https://remote.example','http://192.168.1.2:11434','http://user:secret@127.0.0.1:11434','http://127.0.0.1:11434/path'])
def test_endpoint_is_operator_selected_loopback_only(url):
    with pytest.raises(ValueError):OllamaRuntime(model=MODEL,base_url=url)


@pytest.mark.parametrize('supported,prompt,code',[(False,'text','model_unavailable'),(True,'a'*8192,'context_overflow')])
def test_render_preflight_must_preserve_whole_prompt_and_output_budget(supported,prompt,code):
    runtime,calls,_=runtime_fixture(render_supported=supported,rendered=prompt)
    with pytest.raises(OllamaError,match=code):runtime.generate([{'role':'user','content':'private'}])
    assert sum(p=='/api/chat' for p,_ in calls)==1


@pytest.mark.parametrize('change,code',[
    ({'done_reason':'length'},'output_limit'),({'eval_count':3000},'output_limit'),({'prompt_eval_count':8192},'output_limit'),
    ({'message':{'content':'partial','tool_calls':[{'name':'run'}]}},'generation_failed'),
    ({'message':{'content':'partial','thinking':'private reasoning'}},'generation_failed'),
    ({'remote_host':'https://cloud.example'},'generation_failed'),({'model':'other'},'model_unavailable')])
def test_never_completes_tools_cloud_reasoning_or_truncated_output(change,code):
    final={'model':MODEL,'message':{'content':'Local works'},'done':True,'done_reason':'stop','prompt_eval_count':12,'eval_count':2,**change}
    runtime,_,_=runtime_fixture(stream=[final])
    with pytest.raises(OllamaError,match=code):runtime.generate([{'role':'user','content':'private'}])


def test_cancellation_and_character_limit_never_return_completion():
    runtime,calls,_=runtime_fixture();cancelled=threading.Event();cancelled.set()
    with pytest.raises(OllamaError):runtime.generate([{'role':'user','content':'private'}],cancelled=cancelled)
    assert not calls
    with pytest.raises(OllamaError,match='output_limit'):runtime.generate([{'role':'user','content':'private'}],max_output_characters=2)
