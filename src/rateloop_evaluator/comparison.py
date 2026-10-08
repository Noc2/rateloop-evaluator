"""Private-learning comparisons on one frozen test set, without activating models.

Agreement describes the declared labels and sampling unit. It is not a calibrated
probability, independent-human qualification or a production deployment gate.
"""
from __future__ import annotations

import math
import time
from typing import Any

from .backends import render_input, prepare_inference
from .execution import serialized_training
from .learning import LearningStore, is_independent_reference, _digest
from .protocol import validate_no_demonstration_overlap


def _interval(successes: int, count: int) -> dict | None:
    if count == 0:
        return None
    z = 1.959963984540054
    proportion = successes/count
    denominator = 1+z*z/count
    center = (proportion+z*z/(2*count))/denominator
    half = z*math.sqrt(proportion*(1-proportion)/count+z*z/(4*count*count))/denominator
    return {'estimate': proportion, 'lower': max(0.0, center-half), 'upper': min(1.0, center+half),
            'confidence': .95, 'method': 'wilson', 'sample_count': count}


def label_metrics(confusion: dict[str, dict[str, int]], expected_counts: dict[str, int]) -> dict:
    """Per-label recall includes abstentions; missing classes never imply quality.

    Keep the expected counts separately from confusion because a tie has no
    predicted label. Balanced agreement is defined only when every label occurs.
    These are descriptive metrics over declared reference labels, not confidence.
    """
    labels = set(expected_counts)
    if not labels or set(confusion) != labels:
        raise ValueError('Metrics require the exact declared label set')
    for label, row in confusion.items():
        if (set(row) != labels or type(expected_counts[label]) is not int or expected_counts[label] < 0
                or any(type(count) is not int or count < 0 for count in row.values())
                or sum(row.values()) > expected_counts[label]):
            raise ValueError('Invalid label counts')
    per_label = {}
    for label, support in expected_counts.items():
        correct = confusion[label][label]
        predicted = sum(row[label] for row in confusion.values())
        per_label[label] = {'support': support, 'correct': correct, 'predicted': predicted,
            'abstentions': support-sum(confusion[label].values()),
            'recall': correct/support if support else None,
            'precision': correct/predicted if predicted else None}
    recalls = [row['recall'] for row in per_label.values()]
    return {'per_label': per_label,
        'balanced_agreement': sum(recalls)/len(recalls) if all(value is not None for value in recalls) else None,
        'majority_label_agreement': max(expected_counts.values())/sum(expected_counts.values())
            if sum(expected_counts.values()) else None}


@serialized_training
def compare_snapshot(store: LearningStore, snapshot_id: str, workspace_id: str,
                     models: dict[str, Any], *, now: float | None = None) -> dict:
    """Compare supplied local backends without registration or activation.

    This is dataset validation under private-learning authorization, not a route
    for production inference. Models are provisioned by the trusted caller, never
    supplied as executable uploads. Live permissions are rechecked between rows.
    """
    if not isinstance(models, dict) or not 1 <= len(models) <= 4 or any(not isinstance(key, str) or not key for key in models):
        raise ValueError('Comparison requires one to four named local models')
    snapshot = store.load_snapshot(snapshot_id, workspace_id, now=now)
    if snapshot['purpose'] != 'private_training':
        raise PermissionError('Private comparisons require a private-training snapshot')
    held_out = {row['group_id'] for row in snapshot['test']}
    if held_out & {row['group_id'] for part in ('train', 'calibration') for row in snapshot[part]}:
        raise ValueError('Held-out source groups overlap learning partitions')
    representatives = {}
    # A source group is one sampling unit. Do not inflate a confidence interval
    # by counting repeated revisions or formatting duplicates as new evidence.
    for row in sorted(snapshot['test'], key=lambda row: row['evaluation_id']):
        representatives.setdefault(row['group_id'], row)
    rows = list(representatives.values())
    if not rows:
        raise ValueError('Comparison needs frozen held-out examples')
    results = {}
    for model_id, backend in models.items():
        try:
            question_stats = {q['id']: {'labels': [label['id'] for label in q['labels']],
                'correct': 0, 'count': 0, 'abstentions': 0, 'false_approvals': 0, 'false_rejections': 0,
                'expected_label_counts': {label['id']: 0 for label in q['labels']},
                'confusion': {label['id']: {predicted['id']: 0 for predicted in q['labels']} for label in q['labels']}}
                for q in rows[0]['template']['questions']}
            exact_agreements, durations = 0, []
            for row in rows:
                store.load_snapshot(snapshot_id, workspace_id, now=now)
                questions = row['template']['questions']
                validate_no_demonstration_overlap(row['input']['text'], questions)
                text = render_input(row['input'])
                prepared = prepare_inference(backend, text, questions)
                if prepared.token_count > row['template']['maxTokens']:
                    raise ValueError('Comparison input exceeds the template token limit')
                training = (getattr(backend, 'manifest', None) or {}).get('training') or {}
                selection = training.get('selection') or {}
                if held_out & set(training.get('trainingGroupIds', []) + selection.get('validationGroupIds', [])):
                    raise ValueError('A compared model was trained on the held-out source groups')
                if {example['evaluation_id'] for example in rows} & set(training.get('trainingExampleIds', []) + selection.get('validationExampleIds', [])):
                    raise ValueError('A compared model was trained on the held-out examples')
                started = time.perf_counter()
                scores = prepared.predict()
                durations.append((time.perf_counter()-started)*1000)
                if set(scores) != set(question_stats):
                    raise ValueError('Model scores do not cover the exact criteria')
                all_correct = True
                for question in questions:
                    qid = question['id']
                    stats = question_stats[qid]
                    distribution = scores[qid]
                    if (set(distribution) != set(stats['labels'])
                            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                                   or not math.isfinite(value) or not 0 <= value <= 1 for value in distribution.values())):
                        raise ValueError('Model returned invalid raw label scores')
                    top = max(distribution.values())
                    winners = [key for key, value in distribution.items() if value == top]
                    expected = row['labels'][qid]
                    stats['count'] += 1
                    stats['expected_label_counts'][expected] += 1
                    if len(winners) != 1:
                        stats['abstentions'] += 1
                        all_correct = False
                        continue
                    predicted = winners[0]
                    correct = predicted == expected
                    stats['correct'] += int(correct)
                    all_correct = all_correct and correct
                    stats['confusion'][expected][predicted] += 1
                    passing = set(question.get('passLabels', []))
                    stats['false_approvals'] += int(predicted in passing and expected not in passing)
                    stats['false_rejections'] += int(predicted not in passing and expected in passing)
                exact_agreements += int(all_correct)
            # A revoked permission never produces a usable comparison report.
            store.load_snapshot(snapshot_id, workspace_id, now=now)
            manifest = getattr(backend, 'manifest', None) or {}
            results[model_id] = {'model_manifest_digest': _digest(manifest) if manifest else None,
                'trained': bool(manifest.get('training')),
                'agreement': _interval(exact_agreements, len(rows)),
                'criteria': {qid: {**stats, 'agreement': _interval(stats['correct'], stats['count']),
                                  **label_metrics(stats['confusion'], stats['expected_label_counts']),
                                  'label_coverage': (stats['count']-stats['abstentions'])/stats['count']}
                             for qid, stats in question_stats.items()},
                'mean_prediction_ms': sum(durations)/len(durations)}
        finally:
            # Comparing candidates must not retain multiple full checkpoints.
            if hasattr(backend, "unload"):
                backend.unload()
    provenance = {}
    for row in rows:
        kind = 'independent_human' if is_independent_reference(row) else row.get('label_provenance', 'unverified')
        provenance[kind] = provenance.get(kind, 0)+1
    return {'schema_version': 'rateloop.model-comparison.v1', 'snapshot_id': snapshot_id,
        'snapshot_digest': snapshot['content_digest'], 'workspace_id': workspace_id,
        'test_group_count': len(rows), 'provenance_counts': provenance,
        'independent_reference_count': sum(is_independent_reference(row) for row in rows),
        'models': results, 'quality_gate': False,
        'limits': 'Frozen source-group sample; agreement with declared labels, not a truth probability. Raw scores are uncalibrated. Repeated test-set inspection can bias later model choices.'}
