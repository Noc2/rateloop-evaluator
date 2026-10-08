"""Full-denominator operational reports for frozen English/German cohorts."""
from collections import Counter
from pathlib import Path

from .backends import file_hash
from .general_benchmark import FAMILIES, LANGUAGES, digest
from .general_qualification import STATES as QUALIFICATION_STATES, _percentile, test_representatives

STATES = set(QUALIFICATION_STATES) | {'abstained'}


def implementation_commitment():
    root = Path(__file__).parent
    return digest({path.name: file_hash(path) for path in sorted(root.glob('*.py'))})


def aligned_observations(manifest, rows, observations):
    representatives = test_representatives(manifest, rows)
    expected = {row['evaluation_id'] for _, row in representatives}
    if (not isinstance(observations, list) or len(observations) != len(expected)
            or {item.get('evaluation_id') for item in observations} != expected):
        raise ValueError('Report every frozen test representative exactly once')
    indexed = {item['evaluation_id']: item for item in observations}
    for observation in observations:
        if observation.get('state') not in STATES:
            raise ValueError('Unknown benchmark outcome')
        stages = observation.get('stages_ms', {})
        if not isinstance(stages, dict) or set(stages) - {'load', 'prepare', 'infer'}:
            raise ValueError('Invalid benchmark timing stages')
        value = observation.get('total_ms')
        import math
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid benchmark latency')
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in stages.values()):
            raise ValueError('Invalid benchmark timing stage')
    return [(entry, row, indexed[row['evaluation_id']]) for entry, row in representatives]


def _summary(cases):
    states = Counter(observation['state'] for _, _, observation in cases)
    completed = [o for _, _, o in cases if o['state'] == 'completed']
    all_ms = [o['total_ms'] for _, _, o in cases]
    completed_ms = [o['total_ms'] for o in completed]
    stages = {}
    for stage in ('load', 'prepare', 'infer'):
        values = [o['stages_ms'][stage] for _, _, o in cases if stage in o.get('stages_ms', {})]
        stages[stage] = {'measured_cases': len(values), 'p50': _percentile(values, .5), 'p95': _percentile(values, .95)}
    return {'source_groups': len(cases), 'completed': len(completed), 'states': dict(states),
        'completion_coverage': len(completed)/len(cases) if cases else None,
        'latency_all_ms': {'p50': _percentile(all_ms, .5), 'p95': _percentile(all_ms, .95)},
        'latency_completed_ms': {'p50': _percentile(completed_ms, .5), 'p95': _percentile(completed_ms, .95)},
        'stages_ms': stages}


def full_cohort_report(manifest, rows, observations):
    cases = aligned_observations(manifest, rows, observations)
    def length(row): return sum(len(value) for value in row['input'].values())
    return {'benchmark_commitment': manifest['commitment'], **_summary(cases),
        'languages': {lang: _summary([case for case in cases if case[0]['language'] == lang]) for lang in LANGUAGES},
        'families': {family + '/' + lang: _summary([case for case in cases
                     if case[0]['family'] == family and case[0]['language'] == lang])
                     for family in FAMILIES for lang in LANGUAGES},
        'input_characters': {
            'up_to_1000': _summary([case for case in cases if length(case[1]) <= 1000]),
            '1001_to_3000': _summary([case for case in cases if 1000 < length(case[1]) <= 3000]),
            'over_3000': _summary([case for case in cases if length(case[1]) > 3000])},
        'limits': 'Character strata include all input fields, not rubric tokens. All frozen test representatives remain in '
                  'the denominator. Completion coverage is not decision coverage or calibrated confidence. '
                  'Completed-only latency is secondary to full-cohort latency; fast overflow rejections are separate.'}


def matched_cohort_report(manifest, rows, observations_by_model):
    if not isinstance(observations_by_model, dict) or not 2 <= len(observations_by_model) <= 8:
        raise ValueError('Supply two to eight complete model observations for one frozen benchmark')
    if any(observation.get('benchmark_commitment') != manifest['commitment']
           for observations in observations_by_model.values() for observation in observations):
        raise ValueError('Matched observations must bind the same frozen benchmark')
    cases = {model: aligned_observations(manifest, rows, observations) for model, observations in observations_by_model.items()}
    completed_ids = [{row['evaluation_id'] for _, row, observation in values if observation['state'] == 'completed'}
                     for values in cases.values()]
    shared = set.intersection(*completed_ids)
    return {'schema_version': 'rateloop.matched-cohort-report.v1', 'benchmark_commitment': manifest['commitment'],
        'full_cohort': {model: full_cohort_report(manifest, rows, observations)
                        for model, observations in observations_by_model.items()},
        'common_completed_cases': len(shared),
        'common_completed_latency': {model: _summary([case for case in values if case[1]['evaluation_id'] in shared])
                                     for model, values in cases.items()},
        'qualified': False, 'activation_changed': False,
        'limits': 'Matched completion is a selected secondary cohort, not representative accuracy. '
                  'Do not compare accuracy numbers from different completed subsets; retain each model full-cohort quality report.'}
