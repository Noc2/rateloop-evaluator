from copy import deepcopy
import pytest

from rateloop_evaluator.quality import (balanced_optimizer_examples, group_representatives,
    score_predictions, validation_improved, validation_partition)
from rateloop_evaluator.training import TrainOptions, training_records


def row(index, label=None):
    return {'evaluation_id': f'e{index:04}', 'case_id': f'case{index}', 'group_id': f'group{index}',
        'input': {'text': f'Example {index}', 'context': '', 'evidence': ''},
        'labels': {'q': label or ('yes' if index % 2 else 'no')},
        'template': {'questions': [{'id': 'q', 'text': 'Does it satisfy the criterion?',
            'labels': [{'id': 'yes', 'description': 'Satisfies'}, {'id': 'no', 'description': 'Does not satisfy'}],
            'passLabels': ['yes']}]}}


def snapshot():
    return {'train': [row(i) for i in range(200)], 'calibration': [row(201)], 'test': [row(202)]}


def test_partition_optimizer_consumer_excludes_validation_calibration_test():
    source = snapshot()
    training, validation = validation_partition(source)
    records = training_records(training)
    assert records and validation
    optimizer_texts = {record['input'] for record in records}
    heldout_texts = {record['input'] for record in training_records(validation + source['calibration'] + source['test'])}
    assert optimizer_texts.isdisjoint(heldout_texts)
    reversed_source = {key: list(reversed(value)) for key, value in source.items()}
    assert {r['group_id'] for r in validation_partition(reversed_source)[1]} == {r['group_id'] for r in validation}
    source['train'].extend(row(i) for i in range(300, 500))
    larger = validation_partition(source)[1]
    assert {r['group_id'] for r in larger if int(r['group_id'][5:]) < 200} == {r['group_id'] for r in validation}


@pytest.mark.parametrize('problem', ['group', 'case', 'exact', 'normalized'])
def test_all_holdout_aliases_rejected(problem):
    source = snapshot()
    if problem == 'group': source['test'][0]['group_id'] = source['train'][0]['group_id']
    if problem == 'case': source['test'][0]['case_id'] = source['train'][0]['case_id']
    if problem == 'exact': source['test'][0]['input'] = deepcopy(source['train'][0]['input'])
    if problem == 'normalized': source['test'][0]['input']['text'] = '  EXAMPLE   0  '
    with pytest.raises(ValueError, match='group|duplicate'):
        validation_partition(source)


def test_small_or_missing_class_cannot_silently_select_on_test_labels():
    source = snapshot()
    source['train'] = [row(i, 'yes') for i in range(200)]
    with pytest.raises(ValueError, match='every label'):
        validation_partition(source)
    with pytest.raises(ValueError, match='Insufficient'):
        validation_partition({'train': [row(1)], 'calibration': [row(2)], 'test': [row(3)]})


def test_source_group_is_one_observation_and_ties_are_abstentions():
    rows = [row(1), row(2), row(3)]
    rows[2]['group_id'] = rows[0]['group_id']
    rows = group_representatives(rows)
    assert len(rows) == 2
    result = score_predictions(rows, [{'q': {'yes': .9, 'no': .1}}, {'q': {'yes': .5, 'no': .5}}])
    metrics = result['criteria']['q']
    assert metrics['balanced_agreement'] == .5
    assert metrics['abstentions'] == 1
    assert metrics['brier_score'] == pytest.approx(.26)
    assert metrics['expected_calibration_error'] == pytest.approx(.3)
    assert result['quality_gate'] is False and result['scores_calibrated'] is False


@pytest.mark.parametrize('distribution', [{'yes': .9, 'no': .9}, {'yes': float('nan'), 'no': 0}, {'yes': True, 'no': False}, {'yes': 1}])
def test_metrics_fail_closed_for_invalid_scores(distribution):
    with pytest.raises(ValueError, match='distribution'):
        score_predictions([row(1)], [{'q': distribution}])


@pytest.mark.parametrize('options', [TrainOptions(validation_fraction=.2),
    TrainOptions(validation_fraction=.2, max_steps=20),
    TrainOptions(validation_fraction=.2, max_steps=2001),
    TrainOptions(validation_fraction=.2, max_steps=50, method='full'),
    TrainOptions(validation_fraction=float('nan'), max_steps=50)])
def test_validation_recipe_rejects_unbounded_or_incomplete_runs(options):
    with pytest.raises(ValueError, match='Validation'):
        options.validate()


def test_balancing_uses_optimizer_groups_without_importing_heldouts():
    source = snapshot()
    train, validation = validation_partition(source)
    # Deliberately create imbalance and correlated revisions.
    train = [r for r in train if r['labels']['q'] == 'yes'] + [r for r in train if r['labels']['q'] == 'no'][:6]
    related = deepcopy(train[0]); related['evaluation_id'] += 'revision'
    balanced = balanced_optimizer_examples(train + [related])
    assert sum(r['labels']['q'] == 'yes' for r in balanced) == sum(r['labels']['q'] == 'no' for r in balanced)
    assert {r['group_id'] for r in balanced} == {r['group_id'] for r in train}
    assert {r['group_id'] for r in balanced}.isdisjoint(r['group_id'] for r in validation + source['test'] + source['calibration'])
    assert related['evaluation_id'] not in {r['evaluation_id'] for r in balanced}


def test_validation_gate_rejects_accuracy_illusion_and_class_regressions():
    rows = [row(i, 'yes' if i < 8 else 'no') for i in range(10)]
    def report(predictions):
        return score_predictions(rows, [{'q': {'yes': .9 if p == 'yes' else .1,
            'no': .1 if p == 'yes' else .9}} for p in predictions])
    base = report(['yes']*6+['no']*2+['yes', 'no'])
    majority = report(['yes']*10)
    assert majority['criteria']['q']['agreement'] > base['criteria']['q']['agreement']
    assert validation_improved(base, majority) is False
    assert validation_improved(base, base) is False
    improved = report(['yes']*8+['yes', 'no'])
    assert validation_improved(base, improved) is True
    assert validation_improved(improved, base) is False
