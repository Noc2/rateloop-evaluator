import json
import time

import pytest

from rateloop_evaluator.datasets import import_dataset, load_dataset, preview_dataset, MAX_BYTES
from rateloop_evaluator.learning import LearningStore, provision_key, is_independent_reference
from rateloop_evaluator.protocol import Template, commitment
from rateloop_evaluator.training import training_records

TEMPLATE = {'id': 'summary', 'version': 1, 'language': 'en', 'maxTokens': 512, 'questions': [
    {'id': 'faithful', 'text': 'Does the summary follow the source?', 'labels': [
        {'id': 'yes', 'description': 'Source supports the summary'},
        {'id': 'no', 'description': 'Unsupported claim'}], 'passLabels': ['yes']}]}
FIELDS = ['input.text', 'input.context', 'input.evidence', 'imported_labels']


@pytest.fixture
def store(tmp_path):
    key = tmp_path/'key'
    provision_key(key)
    return LearningStore(tmp_path/'data', key)


def authorize(store, **kw):
    return store.add_grant(workspace_id=kw.pop('workspace_id', 'workspace-a'), rights=['private_training'],
        fields=kw.pop('fields', FIELDS), expires_at=time.time()+3600, evidence='Owner consent', **kw)


def rows(count=8):
    return [{'case_id': f'case-{i}', 'group_id': f'source-{i}', 'text': f'Summary of source number {i}',
             'evidence': f'Source document number {i}', 'label': 'yes' if i % 2 else 'no'} for i in range(count)]


def upload(store, data=None, **kw):
    return import_dataset(store, workspace_id=kw.pop('workspace_id', 'workspace-a'), dataset_id='summaries',
        template=kw.pop('template', TEMPLATE), content='\n'.join(json.dumps(row) for row in (data or rows())),
        format='jsonl', provenance=kw.pop('provenance', 'owner'), evidence='Owner supplied labeled examples', **kw)


def test_import_only_needs_training_right_preserves_provenance_and_encrypts(store):
    authorize(store)
    imported = upload(store)
    assert imported['row_count'] == 8
    assert imported['label_counts'] == {'faithful': {'yes': 4, 'no': 4}}
    assert imported['training_eligible'] and not imported['independent_reference']
    loaded = load_dataset(store, imported['id'], 'workspace-a')
    assert loaded['rows'][0]['input']['text'] == rows()[0]['text']
    with store.transaction() as state:
        assert not state['evaluations'] and not state['feedback']
    assert all(b'Summary of source' not in path.read_bytes() for path in store.root.iterdir())
    with pytest.raises(KeyError):
        load_dataset(store, imported['id'], 'workspace-b')


def test_imported_rows_train_but_never_become_independent_references(store):
    authorize(store)
    imported = upload(store, provenance='synthetic')
    snap = store.create_snapshot('workspace-a', 'summary', 1,
        dataset_version_ids=[imported['id']], include_feedback=False)
    assert training_records(snap['train'])
    assert all(not is_independent_reference(row) and row['human_labels'] == []
               for part in ('train', 'calibration', 'test') for row in snap[part])
    with pytest.raises(ValueError, match='provenance'):
        upload(store, provenance='independent_human')


def test_explicit_label_and_template_scope_is_required_and_atomic(store):
    authorize(store, fields=['input.text', 'input.evidence', 'human_labels'])
    with pytest.raises(PermissionError):
        upload(store)
    authorize(store, case_ids=['case-0'])
    with pytest.raises(PermissionError):
        upload(store)
    with store.transaction() as state:
        assert not state.get('datasets') and not state.get('dataset_examples')
    authorize(store, template_commitments=[commitment(TEMPLATE, 'rateloop.evaluator.template.v1')],
              model_bundle_ids=['base'])
    with pytest.raises(PermissionError):
        upload(store)
    assert upload(store, model_bundle_id='base')


def test_immutable_versions_retry_and_selection(store):
    authorize(store)
    first = upload(store)
    assert upload(store) == first
    more = rows(12)
    second = upload(store, more)
    assert second['version'] == 2 and first['id'] != second['id']
    first_snap = store.create_snapshot('workspace-a', 'summary', 1,
        dataset_version_ids=[first['id']], include_feedback=False)
    second_snap = store.create_snapshot('workspace-a', 'summary', 1,
        dataset_version_ids=[second['id']], include_feedback=False)
    for part in ('train', 'calibration', 'test'):
        assert {r['case_id'] for r in first_snap[part]} <= {r['case_id'] for r in second_snap[part]}
    with pytest.raises(KeyError):
        store.create_snapshot('workspace-b', 'summary', 1, dataset_version_ids=[first['id']])
    with pytest.raises(ValueError, match='selected template'):
        store.create_snapshot('workspace-a', 'another', 1, dataset_version_ids=[first['id']])


def test_csv_mapping_and_jsonl_produce_equivalent_rows():
    csv = 'id,source,material,expected\na,source-a,"A summary, with a comma",yes\n'
    preview = preview_dataset(template=TEMPLATE, content=csv, format='csv', mapping={
        'case_id': 'id', 'group_id': 'source', 'text': 'material', 'labels': {'faithful': 'expected'}})
    assert preview['row_count'] == 1
    assert preview['rows'][0]['input']['text'] == 'A summary, with a comma'
    assert preview['rows'][0]['input']['context'] == ''


@pytest.mark.parametrize('content,format', [
    ('text,text\na,b', 'csv'),
    ('case_id,group_id,text,label\na,a,b,yes,extra', 'csv'),
    ('{"case_id":"a","case_id":"b"}', 'jsonl'),
    ('{"case_id":"a","number":NaN}', 'jsonl'),
    ('[]', 'jsonl'),
    ('', 'jsonl'),
    ('x'*(MAX_BYTES+1), 'jsonl'),
])
def test_malformed_import_is_rejected_without_retaining_content(store, content, format):
    authorize(store)
    with pytest.raises((ValueError, UnicodeError)):
        import_dataset(store, workspace_id='workspace-a', dataset_id='bad', template=TEMPLATE,
            content=content, format=format, provenance='owner', evidence='Owner consent')
    with store.transaction() as state:
        assert not state.get('datasets')


def test_invalid_labels_and_ids_do_not_echo_user_content(store):
    authorize(store)
    invalid = rows()
    invalid[-1]['label'] = 'sensitive-invalid-label'
    with pytest.raises(ValueError, match='row 8') as error:
        upload(store, invalid)
    assert 'sensitive' not in str(error.value)
    assert error.value.__cause__ is None
    with store.transaction() as state:
        assert not state.get('datasets')


def test_source_revocation_cannot_be_reauthorized_by_a_new_grant(store):
    source = authorize(store)
    imported = upload(store)
    snap = store.create_snapshot('workspace-a', 'summary', 1, dataset_version_ids=[imported['id']], include_feedback=False)
    store.register_model_lineage('candidate', snap['id'], 'workspace-a')
    store.revoke_grant(source['id'], 'workspace-a')
    authorize(store)
    with pytest.raises(PermissionError):
        load_dataset(store, imported['id'], 'workspace-a')
    with pytest.raises(PermissionError):
        upload(store)
    with pytest.raises(PermissionError):
        store.assert_model_usable('candidate', 'workspace-a')


def test_delete_purges_dataset_material_snapshots_and_invalidates_version(store):
    authorize(store)
    imported = upload(store)
    snap = store.create_snapshot('workspace-a', 'summary', 1, dataset_version_ids=[imported['id']], include_feedback=False)
    row = snap['train'][0]
    store.delete_case('workspace-a', row['case_id'])
    with pytest.raises(PermissionError):
        load_dataset(store, imported['id'], 'workspace-a')
    with pytest.raises(PermissionError):
        store.load_snapshot(snap['id'], 'workspace-a')
    with store.transaction() as state:
        assert all(r['case_id'] != row['case_id'] for r in state['dataset_examples'].values())
        assert all(r['case_id'] != row['case_id'] for part in ('train', 'calibration', 'test') for r in state['snapshots'][snap['id']][part])


def test_formatting_duplicates_stay_together_and_conflicting_labels_reject(store):
    authorize(store)
    examples = rows()
    examples[1].update(text=examples[0]['text'].upper()+'  ', evidence=examples[0]['evidence'], label=examples[0]['label'])
    imported = upload(store, examples)
    snap = store.create_snapshot('workspace-a', 'summary', 1, dataset_version_ids=[imported['id']], include_feedback=False)
    assert any({'case-0', 'case-1'} <= {r['case_id'] for r in snap[part]} for part in ('train', 'calibration', 'test'))
    examples[1]['label'] = 'yes'
    conflict = upload(store, examples)
    with pytest.raises(ValueError, match='conflicting labels'):
        store.create_snapshot('workspace-a', 'summary', 1, dataset_version_ids=[conflict['id']], include_feedback=False)
