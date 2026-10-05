import importlib.util
from pathlib import Path
import json
import pytest

from rateloop_evaluator.general_benchmark import freeze_benchmark
from rateloop_evaluator.general_experiment import prepare_public, run_local, write_private
from test_general_benchmark import row, SOURCES


def test_private_reports_cannot_overwrite_frozen_evidence(tmp_path):
    path = tmp_path/'plan.json'
    write_private(path, {'threshold': .9})
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        write_private(path, {'threshold': .5})
    assert json.loads(path.read_text()) == {'threshold': .9}


def test_public_prepare_keeps_overflow_inputs_and_external_provenance(monkeypatch):
    source = [{'prompt': f'question-{i}', 'response': ('very long ' * 1000 if i == 0 else f'reply-{i}'),
               'helpfulness': i % 5} for i in range(150)]
    monkeypatch.setattr('rateloop_evaluator.general_experiment.read_source', lambda *_: source)
    rows, sources = prepare_public('ignored', maximum_groups=120)
    assert any(len(r['input']['text']) > 9000 for r in rows)
    assert sources['helpsteer2']['excluded']['middle_score'] == 30
    assert sources['helpsteer2']['label_provenance'] == 'external_human'
    assert {r['language'] for r in rows} == {'en'}
    assert all('independentHuman' not in r for r in rows)


def test_local_runner_freezes_calibration_before_test_and_records_failures(tmp_path, monkeypatch):
    rows = [row(i, labels={'judgment': 'approved' if i % 2 else 'rejected'}) for i in range(600)]
    for r in rows:
        r['template']['maxTokens'] = 512
    manifest = freeze_benchmark(rows, SOURCES)
    output = tmp_path/'experiment'
    roles = {e['evaluation_id']: e['role'] for e in manifest['entries']}
    test_answers = {r['input']['text'] for r in rows if roles[r['evaluation_id']] == 'test'}
    class Backend:
        def __init__(self, *_):
            self.manifest = {'files': {'model.safetensors': 'a'*64, 'tokenizer.json': 'b'*64}}
        def load(self): return self
        def count_tokens(self, text, _): return 600 if text.endswith('0') else 10
        def predict(self, text, _):
            if text in test_answers:
                assert (output/'raw-operating-point.json').is_file()
                assert (output/'calibrated-operating-point.json').is_file()
                assert (output/'diagnostic-calibrations.json').is_file()
            return {'judgment': {'approved': .9, 'rejected': .1}}
    monkeypatch.setattr('rateloop_evaluator.general_experiment.GLiNERBackend', Backend)
    monkeypatch.setattr('rateloop_evaluator.general_experiment.render_input', lambda r:r['text'])
    monkeypatch.setattr('rateloop_evaluator.general_experiment.importlib.metadata.version', lambda _: 'test')
    report = run_local(manifest, rows, model_dir='unused', output=output)
    scope = next(iter(report['raw']['slices'].values()))
    assert scope['states']['overflow'] > 0
    assert report['calibrated'] and report['qualified'] is False
    assert report['raw']['evaluated_cases'] == len(test_answers)
    assert all('input' not in obs for obs in json.loads((output/'raw-observations.json').read_text()))


def test_cli_freeze_packet_point_report_are_executable(tmp_path, monkeypatch, capsys):
    path = Path(__file__).resolve().parents[1]/'scripts/run_general_benchmark.py'
    spec = importlib.util.spec_from_file_location('general_cli', path)
    cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    rows = [row(i) for i in range(20)]
    (tmp_path/'rows.json').write_text(json.dumps(rows))
    (tmp_path/'sources.json').write_text(json.dumps(SOURCES))
    cli.main(['freeze', '--rows', str(tmp_path/'rows.json'), '--sources', str(tmp_path/'sources.json'), '--output', str(tmp_path/'manifest.json')])
    cli.main(['packet', '--rows', str(tmp_path/'rows.json'), '--manifest', str(tmp_path/'manifest.json'), '--output', str(tmp_path/'packet.json')])
    assert len(json.loads((tmp_path/'packet.json').read_text())['cases']) == 20
    assert 'Answer' not in capsys.readouterr().out
