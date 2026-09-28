"""Validation partitions and descriptive quality metrics, never qualification.

Selection only sees frozen training groups. Calibration and final test groups
remain separate. Public/imported labels cannot grant independent-human status.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import math

from .comparison import _interval, label_metrics
from .learning import _digest, _source_aliases


def validation_partition(snapshot: dict, *, fraction: float = .2,
                         minimum_per_label: int = 5, assignments: dict | None = None) -> tuple[list[dict], list[dict]]:
    """Use a fixed hash assignment, independent of optimizer seeds and row order.

    A group's role stays fixed when new examples are appended. Never rebalance
    sparse classes using calibration/test labels; request more examples instead.
    """
    if fraction != .2 or type(minimum_per_label) is not int or minimum_per_label < 2:
        raise ValueError('Validation uses a fixed 20% fraction and at least two groups per label')
    seen, owners = {}, {}
    partitions = {'train': [], 'validation': []}
    grouped = {}
    for part in ('train', 'calibration', 'test'):
        for row in snapshot[part]:
            group = row['group_id']
            previous = seen.setdefault(group, part)
            if previous != part:
                raise ValueError('A source group overlaps learning partitions')
            for alias in _source_aliases(row):
                owner = owners.setdefault(alias, (part, group))
                if owner != (part, group):
                    raise ValueError('Related or duplicate material spans learning groups')
            if part == 'train':
                grouped.setdefault(group, []).append(row)
    assigned = dict(assignments or {})
    for group, rows in grouped.items():
        aliases = {_digest(alias) for row in rows for alias in _source_aliases(row)}
        known = {assigned[alias] for alias in aliases if alias in assigned}
        if len(known) > 1 or known - {'train', 'validation'}:
            raise ValueError('A source family bridges frozen optimizer and validation groups')
        value = int(hashlib.sha256(('rateloop.validation.v1:' + group).encode()).hexdigest(), 16) / 2**256
        role = next(iter(known)) if known else ('validation' if value < fraction else 'train')
        partitions[role].extend(rows)
        assigned.update({alias: role for alias in aliases})
    for rows in partitions.values():
        representatives = group_representatives(rows)
        if not representatives:
            raise ValueError('Insufficient training data for separate validation groups')
        for question in representatives[0]['template']['questions']:
            counts = Counter(row['labels'][question['id']] for row in representatives)
            if any(counts[label['id']] < minimum_per_label for label in question['labels']):
                raise ValueError('Insufficient training data: train and validation each need at least '
                                 f'{minimum_per_label} source groups for every label')
    if assignments is not None:
        assignments.update(assigned)
    return partitions['train'], partitions['validation']


def freeze_validation_partition(store, snapshot: dict, *, minimum_per_label: int = 5):
    """Persist alias roles across versions, source-family changes and restarts."""
    scope = _digest([snapshot['workspace_id'], snapshot['template_id'],
                     snapshot['template_version'], snapshot['purpose']])
    with store.transaction() as state:
        ledger = state.setdefault('validation_split_manifests', {}).setdefault(scope, {})
        return validation_partition(snapshot, minimum_per_label=minimum_per_label, assignments=ledger)


def group_representatives(rows: list[dict]) -> list[dict]:
    """Count each related source once, without picking representatives by label."""
    groups = {}
    for row in sorted(rows, key=lambda row: row['evaluation_id']):
        groups.setdefault(row['group_id'], row)
    return list(groups.values())


def balanced_optimizer_examples(rows: list[dict]) -> list[dict]:
    """Deterministically balance label combinations from optimizer groups only.

    Related examples count once before resampling. Validation/calibration/test
    callers must never pass their rows here; validation_partition enforces the
    complete split before train_snapshot invokes this helper.
    """
    classes = {}
    for row in group_representatives(rows):
        key = tuple(sorted(row['labels'].items()))
        classes.setdefault(key, []).append(row)
    if not classes:
        raise ValueError('No optimizer examples to balance')
    largest = max(map(len, classes.values()))
    return [group[index % len(group)] for index in range(largest)
            for _, group in sorted(classes.items())]


def validation_improved(baseline: dict, candidate: dict) -> bool:
    """Require balanced gain without trading away any observed class recall."""
    if (baseline['balanced_agreement'] is None or candidate['balanced_agreement'] is None
            or candidate['balanced_agreement'] <= baseline['balanced_agreement']
            or set(baseline['criteria']) != set(candidate['criteria'])):
        return False
    for qid, original in baseline['criteria'].items():
        changed = candidate['criteria'][qid]
        if (changed['false_approvals'] > original['false_approvals']
                or set(changed['per_label']) != set(original['per_label'])):
            return False
        if any(original['per_label'][label]['recall'] is None
               or changed['per_label'][label]['recall'] is None
               or changed['per_label'][label]['recall'] < original['per_label'][label]['recall']
               for label in original['per_label']):
            return False
    return True


def score_predictions(rows: list[dict], distributions: list[dict]) -> dict:
    """Report confusion, abstentions, Brier/ECE and class support for raw scores.

    ECE and Brier describe these supplied labels, not certified probabilities.
    No automatic activation or customer qualification follows from this report.
    """
    if not rows or len(rows) != len(distributions):
        raise ValueError('Quality scoring requires one prediction per source group')
    questions = rows[0]['template']['questions']
    if any(row['template']['questions'] != questions for row in rows):
        raise ValueError('Quality scoring requires one immutable rubric')
    stats = {}
    for question in questions:
        labels = [label['id'] for label in question['labels']]
        qid = question['id']
        confusion = {label: dict.fromkeys(labels, 0) for label in labels}
        counts = dict.fromkeys(labels, 0)
        correct = false_approvals = false_rejections = abstentions = 0
        brier = nll = 0.
        bins = [{'count': 0, 'score_sum': 0., 'correct': 0} for _ in range(10)]
        for row, predictions in zip(rows, distributions):
            if set(predictions) != {q['id'] for q in questions}:
                raise ValueError('Prediction criteria differ from the frozen rubric')
            scores = predictions[qid]
            if (set(scores) != set(labels) or any(type(p) not in (int, float) or
                    not math.isfinite(p) or not 0 <= p <= 1 for p in scores.values()) or
                    abs(sum(scores.values()) - 1) > 1e-5):
                raise ValueError('Quality metrics need the complete normalized label distribution')
            expected = row['labels'][qid]
            if expected not in counts:
                raise ValueError('Reference label differs from the frozen rubric')
            counts[expected] += 1
            maximum = max(scores.values())
            winners = [label for label in labels if scores[label] == maximum]
            prediction = winners[0] if len(winners) == 1 else None
            hit = prediction == expected
            correct += hit
            abstentions += prediction is None
            if prediction is not None:
                confusion[expected][prediction] += 1
                passing = set(question.get('passLabels', []))
                false_approvals += prediction in passing and expected not in passing
                false_rejections += prediction not in passing and expected in passing
            brier += sum((scores[label] - int(label == expected))**2 for label in labels)
            nll -= math.log(max(scores[expected], 1e-12))
            bucket = bins[min(9, int(maximum * 10))]
            bucket['count'] += 1; bucket['score_sum'] += maximum; bucket['correct'] += hit
        count = len(rows)
        stats[qid] = {'count': count, 'correct': correct, 'agreement': correct/count,
            'agreement_interval': _interval(correct, count),
            'expected_label_counts': counts, 'confusion': confusion,
            'abstentions': abstentions, 'false_approvals': false_approvals,
            'false_rejections': false_rejections, **label_metrics(confusion, counts),
            'brier_score': brier/count, 'negative_log_likelihood': nll/count,
            'expected_calibration_error': sum(abs(b['correct']-b['score_sum']) for b in bins)/count,
            'calibration_bins': [{'lower': index/10, 'upper': (index+1)/10,
                'count': b['count'], 'mean_raw_score': b['score_sum']/b['count'] if b['count'] else None,
                'accuracy': b['correct']/b['count'] if b['count'] else None} for index,b in enumerate(bins)]}
    balanced = [s['balanced_agreement'] for s in stats.values()]
    return {'group_count': len(rows), 'criteria': stats,
        'balanced_agreement': sum(balanced)/len(balanced) if all(v is not None for v in balanced) else None,
        'mean_brier_score': sum(s['brier_score'] for s in stats.values())/len(stats),
        'quality_gate': False, 'scores_calibrated': False}
