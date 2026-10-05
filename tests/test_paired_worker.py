import json
from pathlib import Path
import plistlib
from types import SimpleNamespace

from rateloop_evaluator import cli
from rateloop_evaluator.enrollment import worker_arguments
from rateloop_evaluator import paired_worker


def test_generation_only_start_and_launchd_use_same_private_paired_configuration(tmp_path,monkeypatch):
    root=tmp_path/'state';cli.run(SimpleNamespace(command='init',state_dir=str(root),workspace='workspace-test'))
    cli.write_private(root/'connector.json',{'baseUrl':'https://rateloop.example','apiKey':'secret-private'})
    model={'model':'qwen3.5:4b','contextTokens':8192}
    cli.write_private(root/'worker.json',{'schemaVersion':'rateloop.local-worker.v1','workerId':'worker-test','modelBundleIds':[],
        'device':'cpu','pollSeconds':5,'generation':{'baseUrl':'http://127.0.0.1:11434','model':model}})
    events=[]
    class Runtime:
        def __init__(self,**kwargs):events.append(('runtime',kwargs))
    class Worker:
        def __init__(self,**kwargs):events.append(('worker',kwargs['model']))
        def run_once(self):return {'state':'idle'}
        def close(self):events.append(('closed',))
    monkeypatch.setattr(paired_worker,'OllamaRuntime',Runtime)
    monkeypatch.setattr(paired_worker,'GenerationWorker',Worker)
    args=worker_arguments(root,once=True)
    assert paired_worker.run_paired(args)=={'generation':{'state':'idle'}}
    assert events[-1]==('closed',)
    monkeypatch.setattr('sys.platform','darwin')
    plist=tmp_path/'worker.plist';args.output=str(plist)
    result=paired_worker.install_paired(args)
    installed=plistlib.loads(plist.read_bytes())
    assert installed['ProgramArguments'][-1]=='start'
    assert 'secret-private' not in json.dumps(installed)
    assert result['outboundOnly'] is True
