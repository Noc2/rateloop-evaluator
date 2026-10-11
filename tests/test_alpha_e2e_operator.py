"""Acceptance-driver checks against real calibration, rights and registry consumers."""
from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
import json
import time

import pytest

from rateloop_evaluator import backends, cli
from rateloop_evaluator.feedback_cycle import import_feedback_snapshot
from rateloop_evaluator.protocol import EvaluationRequest
from test_calibration_snapshot import Backend
from test_feedback_cycle import setup
from test_registry import setup_registry

spec = spec_from_file_location("alpha_e2e_operator", Path(__file__).resolve().parents[1] / "scripts/alpha_e2e_operator.py")
operator = module_from_spec(spec)
spec.loader.exec_module(operator)


def test_small_driver_fixture_observes_actual_calibrator_refusal_without_replacing_it(setup, tmp_path, monkeypatch):
    store, args = setup
    snapshot = import_feedback_snapshot(store, **args)
    load = store.load_snapshot
    def small(*values, **kwargs):
        result = load(*values, **kwargs)
        result['calibration'] = result['calibration'][:1]
        return result
    monkeypatch.setattr(store, 'load_snapshot', small)
    monkeypatch.setattr(cli, 'state', lambda args: (tmp_path, {'workspaceId': 'w'}, store, None))
    monkeypatch.setattr(backends, 'validate_local_model', lambda path: {
        'files': {'model.safetensors': 'a'*64},
        'training': {'bundleId': 'candidate', 'snapshotId': snapshot['id'], 'workspaceId': 'w'}})
    monkeypatch.setattr(backends, 'GLiNERBackend', lambda *args: Backend())
    output = tmp_path/'refused.json'
    report = operator.verify_small_calibration_refusal(tmp_path, snapshot['id'], tmp_path/'model',
        'candidate', output, 'cpu')
    assert report == {'status': 'refused_insufficient_support', 'fitted': False, 'qualityClaim': False}
    assert not output.exists()
    # A supported calibration must not be silently relabelled as the tiny-fixture refusal.
    monkeypatch.setattr(store, 'load_snapshot', load)
    with pytest.raises(RuntimeError, match='Expected acceptance refusal'):
        operator.verify_small_calibration_refusal(tmp_path, snapshot['id'], tmp_path/'model',
            'candidate', output, 'cpu')
    assert output.exists()


def test_refusal_helper_rejects_unrelated_errors_and_unexpected_success(monkeypatch):
    for code, diagnostic in [(0, ''), (1, 'Evaluator: No active private_training grant covers this scope'),
                             (1, 'Evaluator: Model artifact integrity check failed')]:
        def command(args):
            print(diagnostic, file=operator.sys.stderr)
            return code
        monkeypatch.setattr(operator, 'cli', command)
        with pytest.raises(RuntimeError, match='Expected acceptance refusal'):
            operator.expect_refusal(Path('/unused'), 'Insufficient calibration data', 'calibrate')
    def warned(args):
        print('Upstream warning before the terminal policy refusal', file=operator.sys.stderr)
        print('Evaluator: Insufficient calibration data', file=operator.sys.stderr)
        return 1
    monkeypatch.setattr(operator, 'cli', warned)
    operator.expect_refusal(Path('/unused'), 'Insufficient calibration data', 'calibrate')


def test_shadow_activation_needs_exact_live_rights_and_uses_real_registry(setup_registry, tmp_path, monkeypatch):
    store, registry, model, manifest = setup_registry
    body = json.loads((Path(__file__).resolve().parents[1]/'examples/approval-request-en.json').read_text())
    body.update(workspaceId='workspace', modelBundleId='candidate')
    request = EvaluationRequest.model_validate(body)
    manifest.update(template_commitments=[request.template_commitment()], id='base')
    registry.register(manifest, 'workspace', model)
    registry.promote('base', 'workspace', template_commitment=request.template_commitment(), language='en', mode='shadow')
    manifest = deepcopy(manifest); manifest['id'] = 'candidate'
    registry.register(manifest, 'workspace', model)
    monkeypatch.setattr(cli, 'state', lambda args: (tmp_path, {'workspaceId': 'workspace'}, store, registry))
    mode = {'mode': 'shadow'}
    connector = SimpleNamespace(learning=store, sync_grants=lambda: mode)
    with pytest.raises(PermissionError, match='explicit activation'):
        registry.serving_policy('candidate', 'workspace', request.template)
    with pytest.raises(PermissionError, match='No active ai_use'):
        operator.activate_shadow(connector, tmp_path, request)
    grant = store.add_grant(workspace_id='workspace', rights=['ai_use'], expires_at=time.time()+60,
        model_bundle_ids=['candidate'], template_commitments=[request.template_commitment()],
        fields=['input.text','input.context'], evidence='Explicit synthetic test permission')
    mode['mode'] = 'paused'
    with pytest.raises(PermissionError, match='shadow-mode'):
        operator.activate_shadow(connector, tmp_path, request)
    mode['mode'] = 'shadow'
    report = operator.activate_shadow(connector, tmp_path, request)
    assert report['mode'] == 'shadow' and report['qualityClaim'] is False
    assert registry.serving_policy('candidate', 'workspace', request.template)['gate'] is None
    assert registry.get('candidate', 'workspace')['manifest']['calibrations'] == []
    assert registry.rollback('workspace', request.template_commitment(), 'en', expected_bundle_id='base')['bundle_id'] == 'base'
    store.revoke_grant(grant['id'], 'workspace')
    with pytest.raises(PermissionError, match='No active ai_use'):
        operator.activate_shadow(connector, tmp_path, request)
