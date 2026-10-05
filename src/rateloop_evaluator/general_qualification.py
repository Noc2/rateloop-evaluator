"""Coverage-aware diagnostics for a frozen general-request operating point.

This module never signs, registers, qualifies or activates a model. Public imports
remain external diagnostics. Independent promotion still uses ModelRegistry's
trusted snapshot/reference and calibrated held-out evidence path.
"""
from __future__ import annotations

from collections import Counter
import math

from .calibration import false_approval_upper_bound
from .comparison import _interval, label_metrics
from .general_benchmark import digest, verify_manifest
from .quality import score_predictions

STATES = ('completed', 'insufficient_evidence', 'not_checked', 'not_applicable',
          'pending', 'withheld', 'failed', 'unsupported', 'overflow', 'partial')
POINT_SCHEMA = 'rateloop.general-operating-point.v1'


def _finite(value, name, minimum=0, maximum=None):
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f'{name} must be a finite number within its declared bounds')
    return value


def freeze_operating_point(manifest, *, model, route, threshold=.9,
                           calibration_commitment=None, minimum_accepted=200,
                           minimum_coverage=.5, max_false_acceptance=.05):
    verify_manifest(manifest)
    if not isinstance(model, dict) or not all(model.get(k) for k in
            ('weights_sha256', 'tokenizer_sha256', 'adapter', 'runtime', 'quantization')):
        raise ValueError('Pin weights, tokenizer, score adapter, runtime and quantization')
    if not isinstance(route, dict) or not all(route.get(k) for k in ('id', 'version', 'evidence_policy')):
        raise ValueError('Pin the complete route and evidence policy before test inspection')
    for key in ('weights_sha256', 'tokenizer_sha256'):
        if not isinstance(model[key], str) or len(model[key]) != 64 or any(c not in '0123456789abcdef' for c in model[key]):
            raise ValueError('Model/tokenizer commitments must be SHA-256')
    _finite(threshold, 'Threshold', .5, 1)
    _finite(minimum_coverage, 'Coverage target', .5, 1)
    _finite(max_false_acceptance, 'False acceptance target', .000001, .05)
    if type(minimum_accepted) is not int or minimum_accepted < 200:
        raise ValueError('At least 200 independent accepted groups are required per advertised slice')
    if calibration_commitment is not None and (not isinstance(calibration_commitment, str) or not calibration_commitment):
        raise ValueError('A calibrator must have an exact commitment')
    point = {'schema_version': POINT_SCHEMA, 'benchmark_commitment': manifest['commitment'],
        'model': model, 'route': route, 'threshold': threshold,
        'calibration_commitment': calibration_commitment, 'minimum_accepted': minimum_accepted,
        'minimum_coverage': minimum_coverage, 'max_false_acceptance': max_false_acceptance,
        'confidence': .95, 'multiplicity': 'bonferroni_across_observed_criterion_slices',
        'quality_gate': False}
    point['commitment'] = digest(point)
    return point


def verify_operating_point(manifest, point):
    if not isinstance(point, dict) or point.get('schema_version') != POINT_SCHEMA:
        raise ValueError('Unsupported operating point')
    try:
        rebuilt = freeze_operating_point(manifest, **{k: point[k] for k in
            ('model', 'route', 'threshold', 'calibration_commitment', 'minimum_accepted',
             'minimum_coverage', 'max_false_acceptance')})
    except KeyError as exc:
        raise ValueError('Operating point is incomplete') from exc
    if rebuilt != point:
        raise ValueError('Frozen operating point changed after selection')


def test_representatives(manifest, rows):
    verify_manifest(manifest, rows)
    cases = {r['evaluation_id']: r for r in rows}
    representatives = {}
    for entry in manifest['entries']:
        if entry['role'] == 'test':
            key = (entry['family'], entry['language'], entry['template_commitment'], entry['group'])
            # Manifest entries already sort by ID; never select using a label or score.
            representatives.setdefault(key, (entry, cases[entry['evaluation_id']]))
    if not representatives:
        raise ValueError('No frozen test groups; collect more source groups')
    return list(representatives.values())


# Do not let pytest collect this imported helper as a test.
test_representatives.__test__ = False


def _percentile(values, quantile):
    return sorted(values)[max(0, math.ceil(quantile * len(values)) - 1)] if values else None


def score_general_benchmark(manifest, rows, point, observations):
    """Require every frozen test representative, including failures and skips.

    Accept observations only for the predeclared model+route. Lowering a threshold
    requires a new operating-point commitment and fresh holdout, not a UI change.
    """
    verify_operating_point(manifest, point)
    representatives = test_representatives(manifest, rows)
    expected = {r['evaluation_id'] for _, r in representatives}
    if (not isinstance(observations, list) or len(observations) != len(expected)
            or {o.get('evaluation_id') for o in observations} != expected):
        raise ValueError('Report every test representative exactly once, including skipped or failed cases')
    observed = {}
    for observation in observations:
        if observation.get('operating_point_commitment') != point['commitment']:
            raise ValueError('Prediction does not bind the frozen model/route/threshold')
        if observation.get('state') not in STATES:
            raise ValueError('Unknown evaluation state')
        _finite(observation.get('total_ms'), 'End-to-end latency')
        _finite(observation.get('queue_ms', 0), 'Queue latency')
        _finite(observation.get('cost_usd'), 'Full logical evaluation cost')
        if observation.get('queue_ms', 0) > observation['total_ms']:
            raise ValueError('Queue time cannot exceed end-to-end latency')
        if observation['state'] != 'completed' and observation.get('scores'):
            raise ValueError('Skipped or partial checks cannot supply completed-criterion scores')
        observed[observation['evaluation_id']] = observation
    scopes = {}
    for entry, row in representatives:
        for question in row['template']['questions']:
            key = '/'.join((entry['family'], entry['language'], entry['template_commitment'], question['id']))
            scopes.setdefault(key, []).append((entry, row, question, observed[row['evaluation_id']]))
    simultaneous_confidence = 1 - (1 - point['confidence']) / len(scopes)
    slices = {}
    for key, cases in scopes.items():
        question = cases[0][2]
        labels = [label['id'] for label in question['labels']]
        passing = set(question.get('passLabels', []))
        qid = question['id']
        counts = Counter(); expected_counts = dict.fromkeys(labels, 0)
        confusion = {label: dict.fromkeys(labels, 0) for label in labels}
        accepted = wrong = decisions = correct = negative_decisions = false_rejections = 0
        scored_rows, scored_distributions = [], []
        for entry, row, _, observation in cases:
            state = observation['state']; counts[state] += 1
            reference = row['labels'][qid]; expected_counts[reference] += 1
            if state != 'completed':
                continue
            # Reuse the shared strict score/schema validator and proper metrics.
            score_predictions([row], [observation.get('scores', {})])
            scores = observation['scores'][qid]
            scored_rows.append(row); scored_distributions.append(observation['scores'])
            maximum = max(scores.values())
            winners = [label for label, value in scores.items() if value == maximum]
            if len(winners) != 1 or maximum < point['threshold']:
                counts['score_abstention'] += 1
                continue
            prediction = winners[0]
            confusion[reference][prediction] += 1
            decisions += 1; correct += int(prediction == reference)
            accepted += int(prediction in passing)
            wrong += int(prediction in passing and reference not in passing)
            negative_decisions += int(prediction not in passing)
            false_rejections += int(prediction not in passing and reference in passing)
        total = len(cases)
        bound = false_approval_upper_bound(wrong, accepted, point['confidence']) if accepted else None
        joint_bound = false_approval_upper_bound(wrong, accepted, simultaneous_confidence) if accepted else None
        quality = score_predictions(scored_rows, scored_distributions)['criteria'][qid] if scored_rows else None
        if quality:
            for bucket in quality['calibration_bins']:
                correct_in_bin = round(bucket['accuracy'] * bucket['count']) if bucket['count'] else 0
                bucket['accuracy_interval'] = _interval(correct_in_bin, bucket['count'])
        metrics = label_metrics(confusion, expected_counts)
        reasons = []
        if accepted < point['minimum_accepted']:
            reasons.append('insufficient_accepted_source_groups')
        if decisions / total < point['minimum_coverage']:
            reasons.append('insufficient_useful_coverage')
        if bound is None or bound > point['max_false_acceptance']:
            reasons.append('false_acceptance_bound_failed')
        if joint_bound is None or joint_bound > point['max_false_acceptance']:
            reasons.append('simultaneous_bound_failed')
        if any(v['recall'] is None for v in metrics['per_label'].values()):
            reasons.append('missing_reference_class')
        slices[key] = {'family': cases[0][0]['family'], 'language': cases[0][0]['language'],
            'template_commitment': cases[0][0]['template_commitment'], 'criterion': qid,
            'source_groups': total, 'states': dict(counts), 'decisions': decisions,
            'useful_coverage': decisions / total, 'accepted': accepted, 'accepted_coverage': accepted / total,
            'negative_decisions': negative_decisions, 'wrong_acceptances': wrong,
            'false_rejections': false_rejections,
            'false_acceptance': wrong / accepted if accepted else None,
            'false_acceptance_upper_bound': bound, 'simultaneous_false_acceptance_upper_bound': joint_bound,
            'false_positive_rate': wrong / sum(v for k, v in expected_counts.items() if k not in passing)
                if sum(v for k, v in expected_counts.items() if k not in passing) else None,
            'agreement': correct / total, 'agreement_interval': _interval(correct, total),
            'confusion': confusion, **metrics, 'completed_score_diagnostics': quality,
            'score_diagnostics_denominator': len(scored_rows), 'statistical_target_met': not reasons,
            'target_failures': reasons,
            'latency_ms': {'p50': _percentile([c[3]['total_ms'] for c in cases], .5),
                           'p95': _percentile([c[3]['total_ms'] for c in cases], .95)},
            'queue_ms_p95': _percentile([c[3].get('queue_ms', 0) for c in cases], .95),
            'cost_usd': sum(c[3]['cost_usd'] for c in cases)}
    return {'schema_version': 'rateloop.general-benchmark-report.v1',
        'benchmark_commitment': manifest['commitment'], 'operating_point_commitment': point['commitment'],
        'slices': slices, 'sampling': manifest['sampling'], 'confidence': point['confidence'],
        'simultaneous_confidence_per_slice': simultaneous_confidence,
        'evaluated_cases': len(expected), 'cost_usd': sum(o['cost_usd'] for o in observations),
        'quality_gate': False, 'qualified': False, 'activation_changed': False,
        'qualification_blockers': ['external_imports_not_authenticated_blind_references',
            'independent_calibration_and_registry_promotion_required',
            'baseline_per_class_non_regression_required',
            *(['representative_deployment_sampling_required'] if manifest['sampling'] != 'representative_traffic' else [])],
        'limits': 'Fixed imported reference cohort diagnostics only. Scores are not probabilities of truth. '
                  'No benchmark report authorizes confidence display or activation. Repeated holdout inspection needs fresh evidence.'}
