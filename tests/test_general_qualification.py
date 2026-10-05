from copy import deepcopy
import pytest

from rateloop_evaluator.general_benchmark import freeze_benchmark
from rateloop_evaluator.general_qualification import (
    freeze_operating_point, score_general_benchmark, test_representatives,
)
from rateloop_evaluator.calibration import false_approval_upper_bound
from test_general_benchmark import row, SOURCES


def fixture():
    rows = [row(i, labels={'judgment': 'approved' if i % 2 else 'rejected'}) for i in range(1800)]
    manifest = freeze_benchmark(rows, SOURCES)
    point = freeze_operating_point(manifest,
        model={'weights_sha256': 'a'*64, 'tokenizer_sha256': 'b'*64, 'adapter': 'probabilities-v1',
               'runtime': 'python-test', 'quantization': 'float32'},
        route={'id': 'test', 'version': '1', 'evidence_policy': 'supplied-only-v1'})
    observations = []
    for _, r in test_representatives(manifest, rows):
        passing = r['labels']['judgment'] == 'approved'
        observations.append({'evaluation_id': r['evaluation_id'], 'operating_point_commitment': point['commitment'],
            'state': 'completed', 'scores': {'judgment': {'approved': .99 if passing else .01,
                                                       'rejected': .01 if passing else .99}},
            'total_ms': 20, 'queue_ms': 2, 'cost_usd': .001})
    return manifest, rows, point, observations


def test_exact_accepted_denominator_matches_existing_registry_bound_and_never_qualifies():
    manifest, rows, point, observations = fixture()
    observations[0]['scores']['judgment'] = {'approved': .99, 'rejected': .01}
    result = score_general_benchmark(manifest, rows, point, observations)
    scope = next(iter(result['slices'].values()))
    assert scope['false_acceptance_upper_bound'] == false_approval_upper_bound(
        scope['wrong_acceptances'], scope['accepted'])
    assert scope['false_acceptance'] == scope['wrong_acceptances'] / scope['accepted']
    assert scope['accepted'] < scope['source_groups']
    assert scope['useful_coverage'] == 1
    assert scope['completed_score_diagnostics']['calibration_bins'][-1]['accuracy_interval']
    assert result['qualified'] is False and result['quality_gate'] is False
    assert 'external_imports_not_authenticated_blind_references' in result['qualification_blockers']


def test_skips_partial_overflow_and_failures_stay_in_coverage_denominator():
    manifest, rows, point, observations = fixture()
    for i, state in enumerate(('failed', 'partial', 'overflow', 'not_checked', 'insufficient_evidence', 'withheld')):
        observations[i]['state'] = state
        observations[i].pop('scores')
    result = score_general_benchmark(manifest, rows, point, observations)
    scope = next(iter(result['slices'].values()))
    assert scope['source_groups'] == len(observations)
    assert scope['decisions'] == len(observations) - 6
    assert scope['score_diagnostics_denominator'] == len(observations) - 6
    assert scope['states']['overflow'] == 1 and scope['states']['partial'] == 1
    with pytest.raises(ValueError, match='every test'):
        score_general_benchmark(manifest, rows, point, observations[:-1])


def test_ties_and_below_threshold_never_count_as_accepted_or_decided():
    manifest, rows, point, observations = fixture()
    observations[0]['scores']['judgment'] = {'approved': .5, 'rejected': .5}
    observations[1]['scores']['judgment'] = {'approved': .88, 'rejected': .12}
    scope = next(iter(score_general_benchmark(manifest, rows, point, observations)['slices'].values()))
    assert scope['states']['score_abstention'] == 2
    assert scope['decisions'] == len(observations)-2


def test_frozen_threshold_scores_and_operational_measurements_cannot_be_changed():
    manifest, rows, point, observations = fixture()
    bad_point = {**point, 'threshold': .51}
    with pytest.raises(ValueError, match='changed after selection'):
        score_general_benchmark(manifest, rows, bad_point, observations)
    changed = deepcopy(observations)
    changed[0]['operating_point_commitment'] = 'different-route'
    with pytest.raises(ValueError, match='frozen model'):
        score_general_benchmark(manifest, rows, point, changed)
    changed = deepcopy(observations)
    changed[0]['total_ms'] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        score_general_benchmark(manifest, rows, point, changed)
    changed = deepcopy(observations)
    changed[0]['scores']['judgment'] = {'approved': .8, 'rejected': .8}
    with pytest.raises(ValueError, match='normalized'):
        score_general_benchmark(manifest, rows, point, changed)


def test_all_failures_have_zero_coverage_and_no_invented_reliability():
    manifest, rows, point, observations = fixture()
    for obs in observations:
        obs['state'] = 'failed'; obs.pop('scores')
    scope = next(iter(score_general_benchmark(manifest, rows, point, observations)['slices'].values()))
    assert scope['useful_coverage'] == 0
    assert scope['false_acceptance_upper_bound'] is None
    assert scope['completed_score_diagnostics'] is None
    assert scope['statistical_target_met'] is False


def test_inadequate_sample_policy_cannot_reduce_required_evidence():
    manifest, _, point, _ = fixture()
    with pytest.raises(ValueError, match='200'):
        freeze_operating_point(manifest, model=point['model'], route=point['route'], minimum_accepted=59)


def test_insufficient_evidence_judgment_is_not_useful_decision_coverage():
    manifest, rows, point, observations = fixture()
    for r in rows:
        r['template']['questions'][0]['labels'].append({'id': 'insufficient_evidence'})
    manifest = freeze_benchmark(rows, SOURCES)
    point = freeze_operating_point(manifest, model=point['model'], route=point['route'])
    for obs in observations:
        obs['operating_point_commitment'] = point['commitment']
        obs['scores']['judgment'] = {'approved': .01, 'rejected': .01, 'insufficient_evidence': .98}
    scope = next(iter(score_general_benchmark(manifest, rows, point, observations)['slices'].values()))
    assert scope['accepted'] == 0 and scope['decisions'] == 0 and scope['useful_coverage'] == 0
    assert scope['states']['insufficient_evidence_prediction'] == len(observations)
