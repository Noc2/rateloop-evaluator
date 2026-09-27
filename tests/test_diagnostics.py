import json

import pytest

from rateloop_evaluator import cli
from rateloop_evaluator.comparison import label_metrics
from rateloop_evaluator.diagnostics import diagnostic_cases, run_diagnostics
from test_cli import invoke


class Backend:
    device = 'cpu'
    manifest = {'source': {'repository': 'synthetic', 'revision': 'test'}}
    def count_tokens(self, text, questions): return 40
    def predict(self, text, questions): return {'judgment': {'approved': .8, 'rejected': .2}}


def test_bilingual_slices_and_shared_metrics_never_imply_quality():
    cases = diagnostic_cases()
    assert len(cases) == 24 and len({case['id'] for case in cases}) == 24
    result = run_diagnostics(Backend())
    assert result['quality_gate'] is False and result['activation_changed'] is False
    assert result['independent_reference_count'] == 0
    assert len(result['slices']) == 6
    for slice in result['slices'].values():
        assert slice['count'] == 4
        assert slice['balanced_agreement'] == .5
        assert slice['per_label'] == label_metrics(slice['confusion'], slice['expected_label_counts'])['per_label']
        assert slice['prediction_latency_ms']['p95'] >= 0
    assert result['error_ids'] and 'opens at' not in json.dumps(result)


def test_ties_and_invalid_scores_and_no_truncation():
    class Tied(Backend):
        def predict(self, *_): return {'judgment': {'approved': .5, 'rejected': .5}}
    report = run_diagnostics(Tied())
    assert report['overall']['abstentions'] == 24 and report['overall']['balanced_agreement'] == 0
    class Invalid(Backend):
        def predict(self, *_): return {'judgment': {'approved': float('nan'), 'rejected': .2}}
    with pytest.raises(ValueError, match='invalid raw scores'): run_diagnostics(Invalid())
    class Long(Backend):
        def count_tokens(self, *_): return 513
        def predict(self, *_): pytest.fail('Over-budget input was evaluated')
    with pytest.raises(ValueError, match='token budget'): run_diagnostics(Long())


def test_cli_diagnostics_require_no_state_or_permission_mutation(tmp_path, capsys, monkeypatch):
    from rateloop_evaluator import backends
    monkeypatch.setattr(backends, 'GLiNERBackend', lambda *_: Backend())
    output = tmp_path/'report.json'
    result = invoke(capsys, tmp_path/'no-state', 'diagnose', '--model-dir', tmp_path/'synthetic-model', '--output', output)
    assert result['caseCount'] == 24 and result['qualityClaim'] is False
    assert not (tmp_path/'no-state').exists()
    assert json.loads(output.read_text())['label_provenance'] == 'synthetic'


def test_cli_pinned_checkpoint_selection_is_explicit(tmp_path, capsys, monkeypatch):
    from rateloop_evaluator import backends
    calls = []
    def provision(path, **options):
        calls.append(options)
        return {'checkpoint': options['checkpoint']}
    monkeypatch.setattr(backends, 'provision_model', provision)
    for checkpoint in ('base', 'decide'):
        invoke(capsys, tmp_path/'no-state', 'provision', '--model-dir', tmp_path/checkpoint, '--checkpoint', checkpoint)
        assert calls[-1] == {'revision': None, 'checkpoint': checkpoint}
    invoke(capsys, tmp_path/'no-state', 'provision', '--model-dir', tmp_path/'invalid',
           '--backend', 'gliclass', '--checkpoint', 'decide', expected=1)
    assert len(calls) == 2
