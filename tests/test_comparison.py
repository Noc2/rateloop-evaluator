from copy import deepcopy
import pytest

from rateloop_evaluator.comparison import compare_snapshot, label_metrics
from test_datasets import store, authorize, upload


class Backend:
    manifest = {'source': {'repository': 'provisioned-local-base'}}
    def count_tokens(self, text, questions):
        return 30
    def predict(self, text, questions):
        return {'faithful': {'yes': .8, 'no': .2}}


def prepared(store):
    grant = authorize(store)
    dataset = upload(store)
    snapshot = store.create_snapshot('workspace-a', 'summary', 1,
        dataset_version_ids=[dataset['id']], include_feedback=False)
    return grant, snapshot


def test_baseline_candidate_compare_same_frozen_cases_without_activation(store):
    _, snapshot = prepared(store)
    result = compare_snapshot(store, snapshot['id'], 'workspace-a', {'base': Backend(), 'candidate': Backend()})
    assert result['test_group_count'] == len({row['group_id'] for row in snapshot['test']})
    assert result['models']['base']['agreement'] == result['models']['candidate']['agreement']
    assert result['independent_reference_count'] == 0
    assert result['provenance_counts'] == {'owner': result['test_group_count']}
    assert result['quality_gate'] is False
    with store.transaction() as state:
        assert not state['bundles'] and not state['deployments']


def test_training_contamination_is_rejected(store):
    _, snapshot = prepared(store)
    backend = Backend()
    backend.manifest = {'training': {'trainingGroupIds': [snapshot['test'][0]['group_id']]}}
    with pytest.raises(ValueError, match='held-out'):
        compare_snapshot(store, snapshot['id'], 'workspace-a', {'bad': backend})


def test_comparison_rechecks_revocation_between_rows_and_before_return(store):
    grant, snapshot = prepared(store)
    class Revoking(Backend):
        def predict(self, text, questions):
            store.revoke_grant(grant['id'], 'workspace-a')
            return super().predict(text, questions)
    with pytest.raises(PermissionError):
        compare_snapshot(store, snapshot['id'], 'workspace-a', {'revoked': Revoking()})


def test_foreign_workspace_cannot_compare(store):
    _, snapshot = prepared(store)
    with pytest.raises(KeyError):
        compare_snapshot(store, snapshot['id'], 'workspace-b', {'base': Backend()})


@pytest.mark.parametrize('scores', [
    {'faithful': {'yes': float('nan'), 'no': .2}},
    {'faithful': {'yes': .8, 'no': True}},
    {'faithful': {'other': .8, 'no': .2}},
    {'other': {'yes': .8, 'no': .2}},
])
def test_invalid_model_output_fails_closed(store, scores):
    _, snapshot = prepared(store)
    class Invalid(Backend):
        def predict(self, text, questions):
            return scores
    with pytest.raises(ValueError, match='scores|criteria'):
        compare_snapshot(store, snapshot['id'], 'workspace-a', {'invalid': Invalid()})


def test_tied_scores_are_abstentions_not_forced_decisions(store):
    _, snapshot = prepared(store)
    class Tied(Backend):
        def predict(self, text, questions):
            return {'faithful': {'yes': .5, 'no': .5}}
    result = compare_snapshot(store, snapshot['id'], 'workspace-a', {'tie': Tied()})
    criterion = result['models']['tie']['criteria']['faithful']
    assert criterion['abstentions'] == criterion['count']
    assert criterion['label_coverage'] == 0
    assert criterion['false_approvals'] == criterion['false_rejections'] == 0
    assert sum(criterion['expected_label_counts'].values()) == criterion['count']
    assert sum(label['abstentions'] for label in criterion['per_label'].values()) == criterion['count']
    assert all(label['recall'] in (0, None) for label in criterion['per_label'].values())


def test_majority_only_predictions_do_not_hide_minority_failure():
    result = label_metrics({'yes': {'yes': 90, 'no': 0}, 'no': {'yes': 10, 'no': 0}}, {'yes': 90, 'no': 10})
    assert result['majority_label_agreement'] == .9
    assert result['balanced_agreement'] == .5
    assert result['per_label']['no'] == {'support': 10, 'correct': 0, 'predicted': 0,
        'abstentions': 0, 'recall': 0, 'precision': None}


def test_missing_label_and_abstentions_are_visible_in_metrics():
    result = label_metrics({'yes': {'yes': 2, 'no': 0}, 'no': {'yes': 0, 'no': 0}}, {'yes': 4, 'no': 0})
    assert result['balanced_agreement'] is None
    assert result['per_label']['no']['recall'] is None
    assert result['per_label']['yes']['recall'] == .5
    assert result['per_label']['yes']['abstentions'] == 2


@pytest.mark.parametrize('confusion,counts', [({}, {}), ({'yes': {'yes': 2}}, {'yes': 1}),
    ({'yes': {'yes': True}}, {'yes': 1}), ({'yes': {'yes': 1}}, {'no': 1}),
    ({'yes': {'no': 0}}, {'yes': 1})])
def test_invalid_metric_counts_are_rejected(confusion, counts):
    with pytest.raises(ValueError):
        label_metrics(confusion, counts)


def test_existing_snapshot_cannot_score_demonstrations_as_unseen_examples(store, monkeypatch):
    _, snapshot = prepared(store)
    original_load = store.load_snapshot
    def old_snapshot(*args, **kwargs):
        value = deepcopy(original_load(*args, **kwargs))
        for row in value['test']:
            row['template']['questions'][0]['examples'] = [
                {'text': '\u00a0' + row['input']['text'].replace(' ', '\t') + '\ufeff', 'labelId': row['labels']['faithful']}]
        return value
    monkeypatch.setattr(store, 'load_snapshot', old_snapshot)
    class MustNotScore(Backend):
        def predict(self, text, questions):
            pytest.fail('A demonstration must never contribute to held-out agreement')
    with pytest.raises(ValueError, match='demonstration|example'):
        compare_snapshot(store, snapshot['id'], 'workspace-a', {'base': MustNotScore()})


@pytest.mark.parametrize('fail', [False, True])
def test_comparison_releases_each_checkpoint_before_next_model_even_on_failure(store, fail):
    _, snapshot = prepared(store)
    resident=set()
    class Releasing(Backend):
        def __init__(self,name): self.name=name
        def count_tokens(self,*args):
            resident.add(self.name)
            assert len(resident)==1
            return super().count_tokens(*args)
        def predict(self,*args):
            if fail: raise ValueError('Synthetic prediction error')
            return super().predict(*args)
        def unload(self): resident.discard(self.name)
    models={name:Releasing(name) for name in ('base','candidate')}
    if fail:
        with pytest.raises(ValueError,match='Synthetic'): compare_snapshot(store,snapshot['id'],'workspace-a',models)
    else:
        compare_snapshot(store,snapshot['id'],'workspace-a',models)
    assert not resident


@pytest.mark.parametrize('field,value_key',[('validationGroupIds','group_id'),('validationExampleIds','evaluation_id')])
def test_validation_selected_examples_cannot_be_reused_as_final_test(store,field,value_key):
    _,snapshot=prepared(store)
    backend=Backend()
    backend.manifest={'training':{'selection':{field:[snapshot['test'][0][value_key]]}}}
    with pytest.raises(ValueError,match='held-out'):
        compare_snapshot(store,snapshot['id'],'workspace-a',{'contaminated':backend})


def test_identical_full_checkpoints_reuse_predictions_but_different_artifacts_do_not(store, tmp_path, monkeypatch):
    from rateloop_evaluator.backends import GLiNERBackend, PreparedInference, write_model_manifest
    from test_backends import local_artifacts
    _, snapshot = prepared(store)
    path = tmp_path / 'checkpoint'; path.mkdir(); local_artifacts(path)
    other = tmp_path / 'other'; other.mkdir(); local_artifacts(other)
    # Identical names and weights, different tokenizer configuration.
    (other / 'tokenizer_config.json').write_text('{"architecture":"boundary","extra":true}')
    write_model_manifest(other, source={'revision': 'different-tokenizer'})
    calls = []
    monkeypatch.setattr(GLiNERBackend, 'load', lambda self: self)
    def prepare(self, text, questions):
        def predict():
            calls.append(self.model_dir)
            return {'faithful': {'yes': .8, 'no': .2}}
        return PreparedInference(30, predict)
    monkeypatch.setattr(GLiNERBackend, 'prepare', prepare)
    result = compare_snapshot(store, snapshot['id'], 'workspace-a', {
        'baseline': GLiNERBackend(path), 'incumbent': GLiNERBackend(path), 'other': GLiNERBackend(other)})
    count = result['test_group_count']
    assert len(calls) == 2 * count
    assert result['models']['incumbent']['prediction_reused_from'] == 'baseline'
    assert result['models']['incumbent']['prediction_count'] == 0
    assert result['models']['incumbent']['mean_prediction_ms'] is None
    assert result['models']['incumbent']['criteria'] == result['models']['baseline']['criteria']
    assert result['models']['other']['prediction_reused_from'] is None


def test_reuse_rechecks_live_permission_and_full_artifact_integrity(store, tmp_path, monkeypatch):
    import rateloop_evaluator.comparison as comparison
    from rateloop_evaluator.backends import GLiNERBackend, PreparedInference
    from test_backends import local_artifacts
    grant, snapshot = prepared(store)
    path = tmp_path / 'checkpoint'; path.mkdir(); local_artifacts(path)
    monkeypatch.setattr(GLiNERBackend, 'load', lambda self: self)
    monkeypatch.setattr(GLiNERBackend, 'prepare', lambda *_: PreparedInference(30, lambda: {'faithful': {'yes': .8, 'no': .2}}))
    identity = comparison._verified_prediction_identity
    checked = []
    def revoke_before_reuse(backend):
        checked.append(True)
        result = identity(backend)
        if len(checked) == 2: store.revoke_grant(grant['id'], 'workspace-a')
        return result
    monkeypatch.setattr(comparison, '_verified_prediction_identity', revoke_before_reuse)
    with pytest.raises(PermissionError):
        compare_snapshot(store, snapshot['id'], 'workspace-a', {'base': GLiNERBackend(path), 'incumbent': GLiNERBackend(path)})
    (path / 'model.safetensors').write_text('modified after registration')
    with pytest.raises(ValueError, match='integrity'):
        identity(GLiNERBackend(path))


def test_prediction_reuse_identity_is_bound_to_loaded_checkpoint_not_just_disk(store, tmp_path, monkeypatch):
    from rateloop_evaluator.backends import GLiNERBackend, PreparedInference
    from rateloop_evaluator.comparison import _verified_prediction_identity
    from test_backends import local_artifacts
    _, snapshot = prepared(store)
    path = tmp_path / 'checkpoint'; path.mkdir(); manifest_path = local_artifacts(path)
    backend = GLiNERBackend(manifest_path)
    backend._model = object()
    backend.manifest = {'files': {'model.safetensors': 'old-loaded-identity'}}
    with pytest.raises(ValueError, match='Loaded checkpoint differs'):
        _verified_prediction_identity(backend)
    backend._model = None
    monkeypatch.setattr(GLiNERBackend, 'load', lambda self: self)
    def changed_during_prepare(self, *_):
        self.manifest = {**self.manifest, 'source': {'revision': 'changed-during-load'}}
        return PreparedInference(30, lambda: pytest.fail('Changed model must not supply a cached prediction'))
    monkeypatch.setattr(GLiNERBackend, 'prepare', changed_during_prepare)
    with pytest.raises(ValueError, match='identity changed'):
        compare_snapshot(store, snapshot['id'], 'workspace-a', {'baseline': backend})
