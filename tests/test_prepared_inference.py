"""Bind serving, held-out comparisons and benchmarks to one exact preflight."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from rateloop_evaluator import backends
from rateloop_evaluator.backends import GLiNERBackend, PreparedInference, prepare_inference, validate_scores
from rateloop_evaluator.comparison import compare_snapshot
from rateloop_evaluator.general_experiment import _predict
from test_backends import QUESTIONS
from test_comparison import prepared as snapshot_fixture
from rateloop_evaluator.learning import LearningStore, provision_key


@pytest.fixture
def store(tmp_path):
    key = tmp_path / "dataset-key"
    provision_key(key)
    return LearningStore(tmp_path / "dataset-learning", key)
from test_service import setup


def fake_gliner(monkeypatch, token_count=512):
    torch = pytest.importorskip('torch')
    runtime = pytest.importorskip('gliner2.inference.runtime')
    class Batch:
        attention_mask = torch.ones((1, token_count), dtype=torch.int64)
        def __len__(self): return 1
        def to(self, *_): return self
    class Processor:
        calls = 0
        def change_mode(self, **kwargs): assert kwargs == {'is_training': False}
        def collate_fn_inference(self, dataset, **kwargs):
            assert kwargs['max_len'] is None
            assert dataset[0][0] == 'same input'
            self.calls += 1
            return Batch()
    class Model(runtime.ExtractorRuntimeMixin):
        architecture = 'boundary'
        processor = Processor()
        config = SimpleNamespace(max_len=512)
        encoder = SimpleNamespace(config=SimpleNamespace(position_biased_input=False))
        def eval(self): return self
        def parameters(self): yield torch.zeros(1)
        def _build_schema_dicts_and_metadata(self, schemas): return schemas, [{'classification_tasks': ['tone']}]
        def _extract_from_batch(self, batch, threshold, metadata, confidence, spans):
            assert len(batch) == 1 and threshold == .5 and confidence and not spans
            assert metadata == [{'classification_tasks': ['tone']}]
            return [{'tone': [{'label': 'yes', 'confidence': .9}, {'label': 'no', 'confidence': .1}]}]
        def format_results(self, result, confidence, relations, tasks):
            assert confidence and relations == [] and tasks == ['tone']
            return result
    monkeypatch.setattr(backends, 'question_schema', lambda questions: deepcopy(questions))
    backend = GLiNERBackend('/unused'); backend._model = Model()
    return backend


def test_prepared_batch_matches_pinned_public_decoder_with_one_collation(monkeypatch):
    backend = fake_gliner(monkeypatch)
    public = backend._model.extract('same input', QUESTIONS, include_confidence=True, max_len=None)
    backend._model.processor.calls = 0
    call = prepare_inference(backend, 'same input', QUESTIONS)
    assert call.token_count == 512
    assert call.predict() == validate_scores(public, QUESTIONS)
    assert backend._model.processor.calls == 1
    assert set(call.stages_ms) == {'load', 'prepare', 'infer'}
    assert all(value >= 0 for value in call.stages_ms.values())
    backend._model = None
    with pytest.raises(ValueError, match='unloaded'):
        call.predict()


def test_exact_model_limit_rejects_overflow_without_decoding(monkeypatch):
    backend = fake_gliner(monkeypatch, 513)
    call = prepare_inference(backend, 'same input', QUESTIONS)
    assert call.token_count == 513 and backend._model.processor.calls == 1
    with pytest.raises(ValueError, match='context limit'):
        call.predict()


@pytest.mark.parametrize('tokens', [512, 513])
def test_all_consumers_use_prepared_tokens_without_second_prediction_path(setup, store, tokens):
    client, body, _, _, service_backend, _ = setup
    calls = []
    def prepare(text, questions):
        def predict():
            calls.append('infer')
            return {q['id']: {label['id']: 1/len(q['labels']) for label in q['labels']} for q in questions}
        calls.append('prepare')
        return PreparedInference(tokens, predict)
    def unexpected(*args): pytest.fail('A consumer repeated preflight or bypassed the prepared input')
    service_backend.prepare = prepare
    service_backend.count_tokens = service_backend.predict = unexpected
    result = client.post('/v1/evaluate', json=body)
    assert result.status_code == 200
    assert result.json()['abstainReason'] == ('input_too_long' if tokens > 512 else 'uncalibrated')
    observation = _predict(service_backend, {'input': body['input'], 'template': body['template'], 'labels': {q['id']: q['labels'][0]['id'] for q in body['template']['questions']}})
    assert observation['state'] == ('overflow' if tokens > 512 else 'completed')
    _, snapshot = snapshot_fixture(store)
    # Dataset fixture template has a larger cap; bind the same boundary locally.
    original = store.load_snapshot
    def limited(*args, **kwargs):
        value = deepcopy(original(*args, **kwargs))
        for row in value['test']: row['template']['maxTokens'] = 512
        return value
    store.load_snapshot = limited
    if tokens > 512:
        with pytest.raises(ValueError, match='token limit'):
            compare_snapshot(store, snapshot['id'], 'workspace-a', {'base': service_backend})
        assert calls == ['prepare'] * 3
    else:
        result = compare_snapshot(store, snapshot['id'], 'workspace-a', {'base': service_backend})
        assert calls.count('prepare') == calls.count('infer') == 2 + result['test_group_count']
