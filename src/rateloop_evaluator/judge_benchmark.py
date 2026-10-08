"""Local label-only challengers on the same frozen cohort; no invented confidence."""
from collections import Counter
from pathlib import Path
import time

from .backends import prepare_inference, render_input
from .benchmark_reporting import aligned_observations, full_cohort_report, implementation_commitment
from .comparison import _interval, label_metrics
from .general_benchmark import digest, verify_manifest
from .general_experiment import write_private
from .general_qualification import test_representatives
from .ollama_judge import OllamaJudge, JudgeInputOverflow, adapter_commitment


def score_label_only(manifest, rows, point, observations):
    expected_point = {key: value for key, value in point.items() if key != 'commitment'}
    if (point.get('schema_version') != 'rateloop.label-only-operating-point.v1'
            or point.get('benchmark_commitment') != manifest['commitment']
            or point.get('score_type') != 'label_only'
            or point.get('commitment') != digest(expected_point)):
        raise ValueError('Label-only observations require their frozen model and cohort')
    cases = aligned_observations(manifest, rows, observations)
    slices = {}
    for entry, row, observation in cases:
        if observation.get('operating_point_commitment') != point['commitment']:
            raise ValueError('Prediction changed the frozen label-only model')
        if any(key in observation for key in ('scores', 'probabilities', 'confidence')):
            raise ValueError('Label-only results cannot claim numeric confidence')
        labels = observation.get('labels')
        if observation['state'] == 'completed':
            if (not isinstance(labels, dict) or set(labels) != set(row['labels']) or any(
                    labels[q['id']] not in [label['id'] for label in q['labels']] for q in row['template']['questions'])):
                raise ValueError('Judge must return every exact declared label')
        elif labels is not None:
            raise ValueError('Incomplete judgment cannot report completed labels')
        for question in row['template']['questions']:
            qid = question['id']; declared = [label['id'] for label in question['labels']]
            key = '/'.join((entry['family'], entry['language'], entry['template_commitment'], qid))
            stats = slices.setdefault(key, {'family': entry['family'], 'language': entry['language'],
                'template_commitment': entry['template_commitment'], 'criterion': qid, 'source_groups': 0,
                'states': Counter(), 'correct': 0, 'decisions': 0, 'accepted': 0, 'false_approvals': 0,
                'expected_counts': dict.fromkeys(declared, 0),
                'confusion': {label: dict.fromkeys(declared, 0) for label in declared}})
            expected = row['labels'][qid]
            stats['source_groups'] += 1; stats['states'][observation['state']] += 1
            stats['expected_counts'][expected] += 1
            if labels is None: continue
            predicted = labels[qid]
            if predicted == 'insufficient_evidence':
                stats['states']['label_abstention'] += 1
                continue
            stats['decisions'] += 1
            stats['correct'] += int(predicted == expected)
            stats['accepted'] += int(predicted in question.get('passLabels', []))
            stats['false_approvals'] += int(predicted in question.get('passLabels', []) and expected not in question.get('passLabels', []))
            stats['confusion'][expected][predicted] += 1
    for stats in slices.values():
        total = stats['source_groups']
        stats.update(label_metrics(stats['confusion'], stats.pop('expected_counts')))
        stats.update(agreement=stats['correct']/total, agreement_interval=_interval(stats['correct'], total),
                     decision_coverage=stats['decisions']/total, accepted_coverage=stats['accepted']/total,
                     false_acceptance=stats['false_approvals']/stats['accepted'] if stats['accepted'] else None)
        stats['states'] = dict(stats['states'])
    return {'schema_version': 'rateloop.label-only-benchmark-report.v1',
        'benchmark_commitment': manifest['commitment'], 'operating_point_commitment': point['commitment'],
        'slices': slices, 'cohort': full_cohort_report(manifest, rows, observations),
        'calibrated_confidence': False, 'quality_gate': False, 'qualified': False, 'activation_changed': False,
        'limits': 'Agreement with imported labels on the full frozen test denominator. Labels are not probabilities. '
                  'No temperature, ECE, Brier score, confidence threshold, production qualification or activation is inferred.'}


def run_local_judge(manifest, rows, *, model_dir, output, timeout_seconds=60):
    """Uses only an already provisioned, pinned loopback Ollama judge bundle."""
    verify_manifest(manifest, rows)
    if type(timeout_seconds) not in (int, float) or not 0 < timeout_seconds <= 180:
        raise ValueError('Choose a bounded judge timeout up to 180 seconds')
    output = Path(output).resolve(); output.mkdir(parents=True, mode=0o700, exist_ok=False)
    backend = OllamaJudge(model_dir)
    try:
        started = time.perf_counter(); identity = backend.load()
        load_ms = (time.perf_counter()-started)*1000
        point = {'schema_version': 'rateloop.label-only-operating-point.v1',
            'benchmark_commitment': manifest['commitment'], 'score_type': 'label_only',
            'model': identity, 'adapter_commitment': adapter_commitment(identity),
            'implementation_commitment': implementation_commitment(), 'timeout_seconds': timeout_seconds,
            'input_policy': 'all fields and exact rubric; no truncation; no external retrieval'}
        point['commitment'] = digest(point)
        write_private(output/'judge-operating-point.json', point)
        observations = []
        for _, row in test_representatives(manifest, rows):
            began = time.perf_counter(); prepared = None
            observation = {'evaluation_id': row['evaluation_id'], 'operating_point_commitment': point['commitment'],
                'benchmark_commitment': manifest['commitment'], 'state': 'failed', 'queue_ms': 0, 'cost_usd': 0}
            try:
                prepared = prepare_inference(backend, render_input(row['input']), row['template']['questions'])
                observation['token_count'] = prepared.token_count
                if prepared.token_count > row['template']['maxTokens']:
                    observation['state'] = 'overflow'
                else:
                    labels = prepared.predict(timeout_seconds=timeout_seconds)
                    observation.update(state='completed', labels=labels)
            except JudgeInputOverflow:
                observation['state'] = 'overflow'
            except (ValueError, RuntimeError):
                pass
            observation['total_ms'] = (time.perf_counter()-began)*1000
            observation['stages_ms'] = dict(prepared.stages_ms) if prepared else {}
            observations.append(observation)
        report = score_label_only(manifest, rows, point, observations)
        report.update(runtime_identity_check_ms=load_ms, model=identity, implementation_commitment=point['implementation_commitment'],
            measurement_limits='Local loopback runtime only. No hosted queue, network SLA or hardware cost measurement. '
                               'No models downloaded, no external telemetry, no calibrated confidence.')
        write_private(output/'judge-observations.json', observations)
        write_private(output/'report.json', report)
        return report
    finally:
        backend.unload()
