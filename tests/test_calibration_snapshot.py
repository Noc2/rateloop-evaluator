from copy import deepcopy
import pytest

from rateloop_evaluator.calibration_snapshot import fit_snapshot_calibrations, CalibratedComparisonBackend
from rateloop_evaluator.feedback_cycle import import_feedback_snapshot
from rateloop_evaluator.backends import prepare_inference, render_input
from test_feedback_cycle import setup


class Backend:
    manifest = {}
    def count_tokens(self, text, questions): return 513 if 'overlong' in text else 40
    def predict(self, text, questions): return {q['id']: {'approved': .8, 'rejected': .2} for q in questions}
    def unload(self): self.unloaded = True


def fit(store, snapshot, backend):
    return fit_snapshot_calibrations(store, snapshot['id'], 'w', backend, bundle_id='candidate', weights_sha256='a'*64)


def test_only_calibration_groups_fit_and_same_mapping_reaches_comparison(setup):
    store, args = setup
    snapshot = import_feedback_snapshot(store, **args)
    backend = Backend()
    artifacts, coverage = fit(store, snapshot, backend)
    assert artifacts[0]['example_ids'] == [r['group_id'] for r in sorted(snapshot['calibration'], key=lambda r:r['evaluation_id'])]
    assert coverage == {'calibration_group_count': 30, 'fitted_group_count': 30, 'overflow_group_count': 0, 'quality_gate': False}
    assert backend.unloaded
    wrapper = CalibratedComparisonBackend(backend, {'id':'candidate','calibrations':artifacts})
    row = snapshot['test'][0]
    calibrated = prepare_inference(wrapper, render_input(row['input']), row['template']['questions']).predict()
    assert .5 < calibrated['judgment']['approved'] < .8


def test_calibration_whole_group_overflow_preserves_coverage_and_rechecks_support(setup, monkeypatch):
    store, args = setup
    snapshot = import_feedback_snapshot(store, **args)
    original = store.load_snapshot
    def with_overflow(*args, **kwargs):
        value = original(*args, **kwargs)
        repeated = deepcopy(value['calibration'][0]); repeated['input']['text'] = 'overlong'
        repeated['evaluation_id'] += '-revision'
        value['calibration'].append(repeated)
        return value
    monkeypatch.setattr(store, 'load_snapshot', with_overflow)
    artifacts, coverage = fit(store, snapshot, Backend())
    assert coverage['calibration_group_count'] == 30 and coverage['overflow_group_count'] == 1
    assert len(artifacts[0]['example_ids']) == 29
    def mostly_overflow(*args, **kwargs):
        value = original(*args, **kwargs)
        for row in value['calibration'][:12]: row['input']['text'] = 'overlong'
        return value
    monkeypatch.setattr(store, 'load_snapshot', mostly_overflow)
    with pytest.raises(ValueError, match='Insufficient calibration'):
        fit(store, snapshot, Backend())


def test_calibration_rejects_ai_exposure_and_live_revocation(setup, monkeypatch):
    store, args = setup
    snapshot = import_feedback_snapshot(store, **args)
    original = store.load_snapshot
    def exposed(*args, **kwargs):
        value = original(*args, **kwargs)
        value['calibration'][0]['human_labels'][0]['exposed_to_ai'] = True
        return value
    monkeypatch.setattr(store, 'load_snapshot', exposed)
    with pytest.raises(PermissionError, match='blind human'):
        fit(store, snapshot, Backend())
    monkeypatch.setattr(store, 'load_snapshot', original)
    class Revoking(Backend):
        def predict(self, *values):
            store.revoke_grant(args['grant_id'], 'w')
            return super().predict(*values)
    with pytest.raises(PermissionError): fit(store, snapshot, Revoking())


def test_calibration_only_lineage_is_explicit_and_revocable_without_optimizer_claim(setup):
    store, args = setup
    snapshot = import_feedback_snapshot(store, **args)
    store.register_model_lineage('calibration-candidate', snapshot['id'], 'w', adaptation_kind='calibration_only')
    with store.transaction() as state:
        lineage = state['lineage']['calibration-candidate']
        assert lineage['adaptation_kind'] == 'calibration_only'
        assert 'optimizerSteps' not in lineage
    store.assert_model_usable('calibration-candidate', 'w')
    store.revoke_grant(args['grant_id'], 'w')
    with pytest.raises(PermissionError): store.assert_model_usable('calibration-candidate', 'w')
