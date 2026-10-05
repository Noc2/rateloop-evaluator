from argparse import Namespace
import io
import json
from pathlib import Path
import stat

import httpx
import pytest

from rateloop_evaluator import cli, enrollment
from rateloop_evaluator.enrollment import connect, worker_arguments
from test_cli import model_files

TOKEN = 'rl_pair_' + 'a' * 40
KEY = 'rl_device_' + 'b' * 40
IDENTITY = dict(workspaceId='workspace-test', deviceId='device-test', workerId='worker-test',
                agentId='agent-test', agentVersionId='version-test', capabilities=['evaluation'])


def paired(tmp_path, *, mutate=None, failure=None):
    calls = []
    def handler(request):
        assert request.url == 'https://rateloop.example/api/assurance/v2/evaluations/enroll'
        assert 'authorization' not in request.headers
        value = json.loads(request.content)
        assert value['enrollmentToken'] == TOKEN
        calls.append(value)
        if 'bundles' not in value: return httpx.Response(200, json=IDENTITY)
        if failure == 'lost': raise httpx.ReadTimeout('private upstream body', request=request)
        result = {**IDENTITY, 'apiKey': KEY, 'apiKeyId': 'key-test',
                  'modelBundleIds': [b['modelBundleId'] for b in value['bundles']]}
        if mutate: mutate(result)
        return httpx.Response(200, json=result)
    return dict(state_dir=tmp_path/'state', base_url='https://rateloop.example', enrollment_token=TOKEN,
                model_dir=model_files(tmp_path), device='mps', transport=httpx.MockTransport(handler)), calls


def test_pairing_registers_exact_local_manifests_stores_private_key_and_never_grants_processing(tmp_path):
    args, calls = paired(tmp_path)
    result = connect(**args)
    root = args['state_dir']
    assert len(calls) == 2
    assert {r['language'] for r in calls[1]['bundles']} == {'en','de'}
    assert all(r['scoreCapability']['scoreType'] == 'mutually_exclusive_softmax' for r in calls[1]['bundles'])
    stored = {json.loads(p.read_text())['modelBundleId']: json.loads(p.read_text()) for p in (root/'registrations').iterdir()}
    assert all(stored[r['modelBundleId']] == r for r in calls[1]['bundles'])
    config = json.loads((root/'connector.json').read_text())
    assert config['apiKey'] == KEY
    assert result['processingEnabled'] is False and KEY not in json.dumps(result) and TOKEN not in json.dumps(result)
    for p in root.rglob('*.json'):
        assert stat.S_IMODE(p.stat().st_mode) == 0o600
        assert TOKEN not in p.read_text()
    _, _, store, _ = cli.state(Namespace(state_dir=str(root)))
    with store.transaction() as db: assert not db['grants']
    worker = worker_arguments(root)
    assert worker.worker_id == IDENTITY['workerId'] and worker.device == 'mps'
    assert worker.allow_training is False and worker.training_model_dir is None
    assert worker.bundle_id == result['modelBundleIds']
    before = (root/'connector.json').read_bytes()
    with pytest.raises(ValueError, match='new state directory'): connect(**args)
    assert (root/'connector.json').read_bytes() == before and len(calls) == 2


@pytest.mark.parametrize('field', ['workspaceId', 'workerId', 'agentId', 'modelBundleIds', 'apiKey'])
def test_pairing_rejects_changed_binding_or_invalid_credential_without_retry(tmp_path, field):
    args, calls = paired(tmp_path, mutate=lambda r:r.update({field:['other'] if field=='modelBundleIds' else 'other'}))
    with pytest.raises(RuntimeError, match='Revoke this device'): connect(**args)
    assert len(calls) == 2
    assert json.loads((args['state_dir']/'enrollment.json').read_text())['status'] == 'claiming'
    assert not (args['state_dir']/'worker.json').exists()


def test_lost_claim_response_is_not_retried_and_preserves_recovery_state(tmp_path):
    args, calls = paired(tmp_path, failure='lost')
    with pytest.raises(RuntimeError, match='Revoke this device') as e: connect(**args)
    assert 'private upstream body' not in str(e.value) and len(calls)==2
    assert not (args['state_dir']/'worker.json').exists()


@pytest.mark.parametrize('url', ['http://example.com', 'https://user:pass@example.com', 'https://example.com/path', 'https://example.com?key=secret', 'http://127.0.0.1:3000'])
def test_pairing_rejects_unsafe_origin_before_network_or_state(tmp_path, url):
    args, calls = paired(tmp_path); args['base_url']=url
    with pytest.raises(ValueError): connect(**args)
    assert not calls and not args['state_dir'].exists()


def test_cli_reads_code_only_from_stdin_and_start_routes_private_worker_config(tmp_path, monkeypatch, capsys):
    captured=[]
    monkeypatch.setattr(enrollment,'connect',lambda **kwargs:captured.append(kwargs) or {'status':'connected'})
    monkeypatch.setattr('sys.stdin',io.StringIO(TOKEN+'\n'))
    assert cli.main(['--state-dir',str(tmp_path/'state'),'connect','--model-dir',str(tmp_path/'model'),'--token-stdin'])==0
    assert captured[0]['enrollment_token']==TOKEN
    assert TOKEN not in capsys.readouterr().out
    with pytest.raises(SystemExit): cli.main(['connect','--model-dir','/tmp/model','--token',TOKEN])


def test_worker_config_requires_private_permissions_and_fixed_allowlist(tmp_path):
    args,_=paired(tmp_path);connect(**args)
    path=args['state_dir']/'worker.json'
    path.chmod(0o644)
    with pytest.raises(ValueError): worker_arguments(args['state_dir'])
