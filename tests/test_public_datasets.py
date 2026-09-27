"""Only authored source-shaped fixtures; no public corpus or download in tests."""
import hashlib
import json
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from rateloop_evaluator import cli
from rateloop_evaluator.datasets import preview_dataset
from rateloop_evaluator.learning import is_independent_reference
from rateloop_evaluator.protocol import commitment
from rateloop_evaluator.public_datasets import prepare_public_dataset, write_prepared
from test_cli import initialized, invoke

REVISION = 'a' * 40


def prepare(tmp_path, rows, dataset='helpsteer2', **kwargs):
    path = tmp_path/'source.jsonl'
    raw = ''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows).encode()
    path.write_bytes(raw)
    args = {'file': path, 'dataset': dataset, 'revision': REVISION,
            'source_sha256': hashlib.sha256(raw).hexdigest(), 'source_split': 'train', 'language': 'en'}
    if dataset == 'helpsteer2':
        args.update(score='correctness', positive_min=3, negative_max=1)
    args.update(kwargs)
    return prepare_public_dataset(**args), args


def test_helpsteer_thresholds_provenance_groups_and_reproducible_output(tmp_path):
    rows = [{'prompt': 'Explain source A', 'response': f'Response {i}', 'correctness': i} for i in range(5)]
    prepared, _ = prepare(tmp_path, rows)
    normalized = preview_dataset(template=prepared['template'], content=prepared['examples'], format='jsonl')
    assert normalized['row_count'] == 4 and normalized['group_count'] == 1
    assert normalized['label_counts'] == {'judgment': {'approved': 2, 'rejected': 2}}
    assert prepared['excluded'][0]['source']['correctness'] == 2
    assert prepared['manifest']['excluded_counts'] == {'ambiguous_score': 1}
    assert prepared['manifest']['label_provenance'] == 'external_human'
    assert prepared['manifest']['import_provenance'] == 'owner'
    assert prepared['manifest']['independent_reference'] is False
    assert all(row['group_id'] == row['source_group_id'] for row in map(json.loads, prepared['examples'].splitlines()))
    browser_mapping = {'text': 'text', 'context': 'context', 'evidence': 'evidence', 'case_id': 'case_id',
                       'group_id': 'source_group_id', 'labels': {'judgment': 'label'}}
    browser_preview = preview_dataset(template=prepared['template'], content=prepared['examples'], format='jsonl', mapping=browser_mapping)
    assert normalized['rows'] == browser_preview['rows']
    reversed_prepared, _ = prepare(tmp_path, list(reversed(rows)))
    assert prepared['examples'] == reversed_prepared['examples']
    assert prepared['manifest']['output_sha256'] == reversed_prepared['manifest']['output_sha256']


@pytest.mark.parametrize('changed', [
    {'source_sha256': '0'*64}, {'revision': 'main'}, {'source_split': 'validation'},
    {'positive_min': 1, 'negative_max': 3}, {'score': 'verbosity'}, {'language': 'de'},
    {'max_rows': 2001}, {'max_rows': True},
])
def test_source_pins_holds_and_bounds_fail_closed(tmp_path, changed):
    with pytest.raises(ValueError):
        prepare(tmp_path, [{'prompt': 'Question', 'response': 'Answer', 'correctness': 4}], **changed)


def test_bounded_source_parser_rejects_ambiguous_json(tmp_path, monkeypatch):
    from rateloop_evaluator import public_datasets
    path = tmp_path/'invalid.jsonl'
    for content in (b'{"prompt":"one","prompt":"two"}\n', b'{"correctness":NaN}\n', b'[]\n'):
        path.write_bytes(content)
        with pytest.raises(ValueError): public_datasets._read_source(path, hashlib.sha256(content).hexdigest())
    content = b'{"value":"long source"}\n'; path.write_bytes(content)
    monkeypatch.setattr(public_datasets, 'MAX_SOURCE_BYTES', 10)
    with pytest.raises(ValueError, match='64 MiB'):
        public_datasets._read_source(path, hashlib.sha256(content).hexdigest())


def test_whole_groups_capacity_duplicates_and_conflicts(tmp_path):
    rows = [{'prompt': 'Source one', 'response': f'Response {i}', 'correctness': i} for i in (0, 1, 3, 4)]
    with pytest.raises(ValueError, match='complete source group'):
        prepare(tmp_path, rows, max_rows=3)
    prepared, _ = prepare(tmp_path, rows + [rows[0]])
    assert prepared['manifest']['row_count'] == 4
    assert prepared['manifest']['excluded_counts'] == {'duplicate_input': 1}
    with pytest.raises(ValueError, match='conflicting labels'):
        prepare(tmp_path, rows + [{**rows[0], 'correctness': 4}])


def test_principle_requires_exact_single_language_and_criterion(tmp_path):
    base = {'context': [{'role': 'user', 'content': 'Give a short answer'}],
            'response': 'Short answer', 'principle': 'conciseness', 'language': 'english', 'fulfilment': 'Yes'}
    rows = [base, {**base, 'response': 'An incorrect answer', 'fulfilment': 'No'},
            {**base, 'principle': 'correctness'}, {**base, 'language': 'german'}]
    prepared, _ = prepare(tmp_path, rows, 'helpsteer3-principle', principle='conciseness')
    assert prepared['manifest']['row_count'] == 2
    assert prepared['manifest']['group_count'] == 1
    assert prepared['manifest']['import_provenance'] == 'ai_assisted'
    assert prepared['manifest']['excluded_counts'] == {'other_principle': 1, 'other_language': 1}
    assert 'conciseness' in prepared['template']['questions'][0]['text']
    with pytest.raises(ValueError):
        prepare(tmp_path, [base], 'helpsteer3-principle')
    with pytest.raises(ValueError, match='schema'):
        prepare(tmp_path, [{**base, 'fulfilment': 'maybe'}], 'helpsteer3-principle', principle='conciseness')


def pii(text='Email alex@example.test', value='alex@example.test', language='en', index=1):
    start = text.index(value)
    return {'source_text': text, 'masked_text': text.replace(value, f'[EMAIL_{index}]'),
            'privacy_mask': [{'value': value, 'start': start, 'end': start+len(value), 'label': 'EMAIL'}],
            'language': language, 'split': 'train'}


def test_pii_synthetic_entity_negatives_and_mask_family(tmp_path):
    rows = [pii(), pii('Email sam@example.test', 'sam@example.test', index=9),
            {'source_text': 'Nothing to see.', 'masked_text': 'Nothing to see.', 'privacy_mask': [], 'language': 'en'}]
    prepared, _ = prepare(tmp_path, rows, 'openpii1m', entity='EMAIL')
    assert prepared['manifest']['row_count'] == 3
    assert prepared['manifest']['group_count'] == 2
    assert prepared['manifest']['import_provenance'] == 'synthetic'
    values = [json.loads(line) for line in prepared['examples'].splitlines()]
    emails = [row for row in values if row['label'] == 'approved']
    assert len({row['group_id'] for row in emails}) == 1
    assert all('[EMAIL' not in row['text'] for row in values)
    german, _ = prepare(tmp_path, [pii('Kontakt alex@example.test', language='de')], 'openpii1m', entity='EMAIL', language='de')
    assert german['template']['language'] == 'de'


@pytest.mark.parametrize('alter', [
    lambda row: row.update(masked_text='Unrelated mask'),
    lambda row: row['privacy_mask'][0].update(start=0),
    lambda row: row['privacy_mask'][0].update(value='wrong'),
    lambda row: row['privacy_mask'][0].update(label='UNKNOWN'),
    lambda row: row['privacy_mask'].append(row['privacy_mask'][0]),
])
def test_invalid_pii_spans_do_not_become_negative_examples(tmp_path, alter):
    row = pii(); alter(row)
    with pytest.raises(ValueError, match='schema'):
        prepare(tmp_path, [row], 'openpii1m', entity='EMAIL')


def test_private_files_no_overwrite_and_cli_without_initialization(tmp_path, capsys):
    prepared, args = prepare(tmp_path, [{'prompt': 'Question', 'response': 'Answer', 'correctness': 4}])
    output = tmp_path/'prepared'
    result = invoke(capsys, tmp_path/'no-state', 'prepare-public-dataset', '--file', args['file'],
                    '--dataset', 'helpsteer2', '--revision', REVISION, '--source-sha256', args['source_sha256'],
                    '--source-split', 'train', '--language', 'en', '--score', 'correctness',
                    '--positive-min', '3', '--negative-max', '1', '--output-dir', output)
    assert result['trainingStarted'] is False and not (tmp_path/'no-state').exists()
    assert set(path.name for path in output.iterdir()) == {'examples.jsonl', 'template.json', 'manifest.json', 'excluded.jsonl'}
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in output.iterdir())
    with pytest.raises(ValueError, match='new directory'):
        write_prepared(output, prepared)


def test_import_and_frozen_snapshot_share_source_group_invariant(initialized, tmp_path, capsys):
    rows = [{'prompt': f'Question {group}', 'response': f'Answer {group}-{label}', 'correctness': label}
            for group in range(12) for label in (0, 4)]
    prepared, _ = prepare(tmp_path, rows)
    out = tmp_path/'import'; write_prepared(out, prepared)
    invoke(capsys, initialized, 'grant', '--right', 'private_training', '--template', 'custom-text-evaluation',
           '--field', 'input.text', '--field', 'input.context', '--field', 'imported_labels',
           '--evidence', 'Authored source-shaped fixture, authorized for local test')
    imported = invoke(capsys, initialized, 'dataset-import', '--file', out/'examples.jsonl',
        '--template-file', out/'template.json', '--format', 'jsonl', '--dataset-id', 'fixture-public',
        '--provenance', prepared['manifest']['import_provenance'], '--evidence', 'Authored source-shaped fixture')
    command = ('snapshot', '--template', 'custom-text-evaluation', '--version', '1',
               '--template-commitment', prepared['manifest']['template_commitment'],
               '--dataset-version', imported['id'], '--no-feedback')
    first = invoke(capsys, initialized, *command)
    _, _, store, _ = cli.state(SimpleNamespace(state_dir=initialized))
    snapshot = store.load_snapshot(first['snapshotId'], 'workspace-test')
    group_parts = {}
    for partition in ('train', 'calibration', 'test'):
        for row in snapshot[partition]:
            group_parts.setdefault(row['source_group_id'], set()).add(partition)
            assert not is_independent_reference(row)
    assert len(group_parts) == 12 and all(len(parts) == 1 for parts in group_parts.values())
    second = invoke(capsys, initialized, *command)
    later = store.load_snapshot(second['snapshotId'], 'workspace-test')
    assert {part: {row['case_id'] for row in snapshot[part]} for part in ('train', 'calibration', 'test')} == {
        part: {row['case_id'] for row in later[part]} for part in ('train', 'calibration', 'test')}


def test_dataset_preview_and_import_reject_demonstration_overlap(initialized, tmp_path, capsys):
    from rateloop_evaluator.templates import custom_text_evaluation
    template = custom_text_evaluation('en', 'Does the text state a budget?', 'Yes', 'No',
                                     examples=[{'text': 'Budget: 25 EUR', 'labelId': 'approved'}]).model_dump()
    source = tmp_path/'overlap.jsonl'
    source.write_text(json.dumps({'case_id': 'case', 'group_id': 'group', 'text': '  Budget:   25\tEUR ', 'label': 'approved'}))
    path = tmp_path/'template.json'; path.write_text(json.dumps(template))
    with pytest.raises(ValueError, match='row 1'):
        preview_dataset(template=template, content=source.read_bytes(), format='jsonl')
    for command in ('dataset-preview', 'dataset-import'):
        extra = () if command == 'dataset-preview' else ('--dataset-id', 'overlap', '--provenance', 'owner', '--evidence', 'Synthetic fixture')
        invoke(capsys, initialized, command, '--file', source, '--template-file', path, '--format', 'jsonl', *extra, expected=1)
    _, _, store, _ = cli.state(SimpleNamespace(state_dir=initialized))
    with store.transaction() as state:
        assert not state.get('datasets')
