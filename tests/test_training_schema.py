"""Actual upstream transformations bind training and inference to one rubric."""
from copy import deepcopy
import random

import pytest

pytest.importorskip('gliner2.processor')
from gliner2.processor import SamplingConfig, SchemaTransformer

from rateloop_evaluator.backends import question_schema
from rateloop_evaluator.templates import custom_text_evaluation
from rateloop_evaluator.training import exact_classification_processor, training_records


def processor(training):
    # This transformation uses no model/tokenizer. Exercise the real pinned
    # upstream implementation without downloads or neural test doubles.
    value = SchemaTransformer.__new__(SchemaTransformer)
    value.is_training = training
    value.tokenizer = object()
    value.sampling_config = SamplingConfig()
    return value


def transform(value, schema, sampling=None):
    schema = deepcopy(schema)
    result = value._infer_from_json(schema)
    return schema, result['schemas'], result['structure_labels'], result['task_types']


@pytest.mark.parametrize('expected', ['approved', 'rejected'])
@pytest.mark.parametrize('with_examples', [False, True])
def test_optimizer_and_inference_keep_identical_rubric_and_correct_target_across_augmentation_seeds(expected, with_examples):
    template = custom_text_evaluation('en', 'Should this record use Route North?', 'Route North', 'Route South').model_dump()
    if with_examples:
        template['questions'][0]['examples'] = [
            {'text': 'Example record with code A1.', 'labelId': 'approved'},
            {'text': 'Example record with code B2.', 'labelId': 'rejected'}]
    training = training_records([{'input': {'text': 'A separate synthetic record.'},
        'template': template, 'labels': {'judgment': expected}}])[0]['output']
    inference = question_schema(template['questions']).schema
    ordinary = processor(True)
    fixed = exact_classification_processor(ordinary)
    reference = transform(processor(False), inference)
    for seed in range(64):
        random.seed(seed)
        actual, tokens, labels, types = transform(fixed, training, SamplingConfig())
        target = actual['classifications'][0]
        assert target['labels'] == ['approved', 'rejected']
        assert target['true_label'] == [expected]
        assert [int(label in target['true_label']) for label in target['labels']] == ([1, 0] if expected == 'approved' else [0, 1])
        assert (tokens, labels, types) == reference[1:]
        assert fixed.is_training is True and ordinary.is_training is True
        assert fixed.tokenizer is ordinary.tokenizer
    assert '_infer_from_json' not in ordinary.__dict__


def test_upstream_default_can_encode_the_original_true_label_as_negative_but_fixed_consumer_cannot():
    template = custom_text_evaluation('en', 'Should this record use Route North?', 'Route North', 'Route South').model_dump()
    training = training_records([{'input': {'text': 'Record with code Q7M.'},
        'template': template, 'labels': {'judgment': 'approved'}}])[0]['output']
    random.seed(4)
    corrupted, *_ = transform(processor(True), training, SamplingConfig())
    record = corrupted['classifications'][0]
    assert 'approved' in record['labels'] and 'approved' not in record['true_label']
    random.seed(4)
    corrected, *_ = transform(exact_classification_processor(processor(True)), training, SamplingConfig())
    assert corrected['classifications'][0]['labels'] == ['approved', 'rejected']
    assert corrected['classifications'][0]['true_label'] == ['approved']


def test_multiple_questions_with_prefix_ids_bind_exact_targets_and_preserve_inference_order():
    template = custom_text_evaluation('en', 'Is this the north route?', 'North', 'South').model_dump()
    first = template['questions'][0]
    first['id'] = 'q'
    second = deepcopy(first)
    second.update(id='quality', text='Does the record pass the quality check?',
        labels=[{'id': 'pass', 'description': 'Meets the quality standard'},
                {'id': 'fail', 'description': 'Does not meet the standard'}], passLabels=['pass'],
        examples=[{'text': 'A complete reference.', 'labelId': 'pass'},
                  {'text': 'An incomplete reference.', 'labelId': 'fail'}])
    template['questions'].append(second)
    record = training_records([{'input': {'text': 'A separate synthetic record.'},
        'template': template, 'labels': {'q': 'approved', 'quality': 'fail'}}])[0]['output']
    fixed = exact_classification_processor(processor(True))
    reference = processor(False)._infer_from_json(question_schema(template['questions']).schema)
    for seed in range(64):
        random.seed(seed)
        schema = deepcopy(record)
        processed = fixed._infer_from_json(schema)
        outputs = fixed._build_outputs(processed, schema, [], 0)
        assert processed['schemas'] == reference['schemas']
        assert [output['output'] for output in outputs] == [[1, 0], [0, 1]]
        assert [output['schema_tokens'] for output in outputs] == reference['schemas']
        assert fixed.is_training is True


def test_schema_mode_is_restored_after_failure():
    fixed = exact_classification_processor(processor(True))
    with pytest.raises(KeyError):
        fixed._infer_from_json({'classifications': [{'task': 'missing-labels'}]})
    assert fixed.is_training is True
