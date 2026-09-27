"""One isolation rule binds blind feedback, saved snapshots and qualification."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from rateloop_evaluator import backends, cli
from rateloop_evaluator.learning import _digest, is_independent_reference
from rateloop_evaluator.protocol import commitment, validate_reference_demonstration_isolation
from rateloop_evaluator.registry import BundleRegistry
from rateloop_evaluator.templates import custom_text_evaluation
from test_learning import store, grant
from test_registry import gate_inputs

DEMONSTRATION = 'The budget is EUR 500.'
VARIANTS = (DEMONSTRATION, '  The\tbudget is\nEUR 500. ', '\ufeffThe\u00a0budget\u2028is EUR 500.\ufeff')


def human_rows(store, *, overlap=None):
    grant(store)
    template = custom_text_evaluation('en', 'Does the text state a numeric budget?', 'Yes', 'No',
                                     [{'text': DEMONSTRATION, 'labelId': 'approved'}]).model_dump()
    template_digest = commitment(template, 'rateloop.evaluator.template.v1')
    for index in range(12):
        text = overlap if index == 0 and overlap is not None else f'The budget for project {index} is EUR {1000+index}.'
        payload = {'text': text, 'context': '', 'evidence': ''}
        evaluation = store.record_evaluation(evaluation_id=f'eval-{index}', workspace_id='workspace-a',
            case_id=f'case-{index}', input_commitment=commitment(payload, 'test.input'),
            template_commitment=template_digest, template=template, input_payload=payload,
            group_id=f'source-{index}', fields=['input.text'])
        store.add_feedback(workspace_id='workspace-a', evaluation_id=evaluation['evaluation_id'],
            input_commitment=evaluation['input_commitment'], template_commitment=template_digest,
            annotator_id='independent-reviewer', labels={'judgment': 'approved'},
            exposed_to_ai=False, independent_human=True)
    return template


def saved_snapshot(store, partition, text):
    human_rows(store)
    snapshot = store.create_snapshot('workspace-a', 'custom-text-evaluation', 1)
    # Simulate an authentic historical snapshot created before the new rule.
    # Its digest remains correct, so rejection must come from isolation, not a
    # generic tampering check. Human labels are independently collected.
    with store.transaction() as state:
        historical = state['snapshots'][snapshot['id']]
        historical[partition][-1]['input']['text'] = text
        historical['content_digest'] = _digest({key: historical[key] for key in
            ('workspace_id', 'template_id', 'template_version', 'purpose', 'train', 'calibration', 'test')})
        result = deepcopy(historical)
    assert all(is_independent_reference(row) for part in ('calibration', 'test') for row in result[part])
    return result


@pytest.mark.parametrize('text', VARIANTS)
def test_blind_human_feedback_does_not_make_a_disclosed_example_held_out(store, text):
    human_rows(store, overlap=text)
    with pytest.raises(ValueError, match='repeat a rubric demonstration'):
        store.create_snapshot('workspace-a', 'custom-text-evaluation', 1)
    with store.transaction() as state:
        assert not state['snapshots']
        assert not state.get('split_manifests')
        assert len(state['feedback']) == 12
        assert all(not row['quarantine_reasons'] for row in state['feedback'].values())


@pytest.mark.parametrize('partition', ('train', 'calibration', 'test'))
@pytest.mark.parametrize('text', VARIANTS)
def test_historical_snapshot_load_checks_every_partition(store, partition, text):
    snapshot = saved_snapshot(store, partition, text)
    with pytest.raises(ValueError, match='repeat a rubric demonstration'):
        store.load_snapshot(snapshot['id'], 'workspace-a')


@pytest.mark.parametrize('command,partition', [('calibrate', 'calibration'), ('score-test', 'test')])
@pytest.mark.parametrize('text', VARIANTS)
def test_calibration_and_score_test_defend_against_disclosed_blind_references(store, tmp_path, monkeypatch, command, partition, text):
    snapshot = saved_snapshot(store, partition, text)
    # Bypass only loading to bind the CLI's additional check to the same rule.
    source = SimpleNamespace(load_snapshot=lambda *_: deepcopy(snapshot))
    registry = SimpleNamespace(get=lambda *_: {'artifact_root': str(tmp_path/'model'),
        'manifest': {'snapshot_id': snapshot['id'], 'selective_policy': {'threshold': .95}, 'max_tokens': 512}})
    monkeypatch.setattr(cli, 'state', lambda _: (tmp_path, {'workspaceId': 'workspace-a'}, source, registry))
    monkeypatch.setattr(backends, 'validate_local_model', lambda _: {'training': {
        'bundleId': 'candidate', 'snapshotId': snapshot['id'], 'workspaceId': 'workspace-a'}})
    monkeypatch.setattr(backends, 'GLiNERBackend', lambda *_: pytest.fail('Disclosed reference reached model inference'))
    args = SimpleNamespace(command=command, _model_execution_owned=True, snapshot_id=snapshot['id'],
        model_dir=tmp_path/'model', bundle_id='candidate', device='cpu', output=tmp_path/'evidence.json', valid_hours=24)
    with pytest.raises(ValueError, match='repeat a rubric demonstration'):
        cli.run(args)
    assert not args.output.exists()


@pytest.mark.parametrize('partition', ('train', 'calibration', 'test'))
@pytest.mark.parametrize('text', VARIANTS)
def test_final_qualification_cannot_accept_disclosed_answers(partition, text):
    manifest, snapshot, evidence = gate_inputs()
    question = {'id': 'q', 'text': 'Does this state a budget?', 'labels': [
        {'id': 'yes', 'description': 'Yes'}, {'id': 'no', 'description': 'No'}], 'passLabels': ['yes'],
        'examples': [{'text': DEMONSTRATION, 'labelId': 'yes'}]}
    for part in ('train', 'calibration', 'test'):
        for index, row in enumerate(snapshot[part]):
            row['template'] = {'language': 'en', 'questions': [question]}
            row['input'] = {'text': f'A separate {part} artifact {index}', 'context': '', 'evidence': ''}
    # Valid evidence otherwise passes; missing one criterion's source isolation
    # must not silently turn it into a probability or automation qualification.
    assert BundleRegistry._quality_gate(manifest, snapshot, evidence)['auto_approvals'] == 300
    snapshot[partition][-1]['input']['text'] = text
    with pytest.raises(ValueError, match='repeat a rubric demonstration'):
        validate_reference_demonstration_isolation(snapshot[partition])
    with pytest.raises(ValueError, match='repeat a rubric demonstration'):
        BundleRegistry._quality_gate(manifest, snapshot, evidence)


def test_nonrepresentative_group_member_cannot_hide_disclosed_reference(store, tmp_path, monkeypatch):
    human_rows(store)
    snapshot = store.create_snapshot('workspace-a', 'custom-text-evaluation', 1)
    duplicate = deepcopy(snapshot['test'][0])
    duplicate.update(evaluation_id='zz-last-in-group', input={'text': VARIANTS[1], 'context': '', 'evidence': ''})
    snapshot['test'].append(duplicate)
    source = SimpleNamespace(load_snapshot=lambda *_: deepcopy(snapshot))
    registry = SimpleNamespace(get=lambda *_: {'artifact_root': str(tmp_path/'model'),
        'manifest': {'snapshot_id': snapshot['id'], 'selective_policy': {'threshold': .95}, 'max_tokens': 512}})
    monkeypatch.setattr(cli, 'state', lambda _: (tmp_path, {'workspaceId': 'workspace-a'}, source, registry))
    monkeypatch.setattr(backends, 'GLiNERBackend', lambda *_: pytest.fail('Nonrepresentative disclosed reference was ignored'))
    with pytest.raises(ValueError, match='repeat a rubric demonstration'):
        cli.run(SimpleNamespace(command='score-test', _model_execution_owned=True, bundle_id='candidate',
            device='cpu', output=tmp_path/'evidence.json', valid_hours=24))


def test_legacy_no_example_rows_keep_their_previous_qualification_behavior():
    manifest, snapshot, evidence = gate_inputs()
    validate_reference_demonstration_isolation(row for part in ('train', 'calibration', 'test') for row in snapshot[part])
    assert BundleRegistry._quality_gate(manifest, snapshot, evidence)['auto_approvals'] == 300
