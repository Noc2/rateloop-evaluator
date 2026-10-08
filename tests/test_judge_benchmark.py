from copy import deepcopy
import json

import pytest

from rateloop_evaluator.benchmark_reporting import full_cohort_report, matched_cohort_report
from rateloop_evaluator.general_benchmark import freeze_benchmark
from rateloop_evaluator.general_qualification import test_representatives
from rateloop_evaluator.judge_benchmark import run_local_judge, score_label_only
from rateloop_evaluator.ollama_judge import JudgeInputOverflow
from rateloop_evaluator.templates import custom_text_evaluation
from test_general_benchmark import SOURCES, row


def bilingual_cohort():
    rows = []
    for i in range(600):
        language = 'en' if i % 2 else 'de'
        item = row(i, language=language, labels={'judgment': 'approved' if i % 3 else 'rejected'})
        item['template'] = custom_text_evaluation(language, 'Meets the criterion?', 'Yes', 'No').model_dump()
        rows.append(item)
    return freeze_benchmark(rows, SOURCES), rows


def test_pinned_label_only_challenger_keeps_full_bilingual_denominator(tmp_path, monkeypatch):
    manifest, rows = bilingual_cohort()
    output = tmp_path/'judge'; events = []
    class Judge:
        def __init__(self, path): assert path == 'explicit-local-bundle'
        def load(self): return {'model': 'qwen3.5:4b', 'weightDigest': 'sha256:'+'a'*64, 'runtimeVersion': 'test'}
        def unload(self): events.append('closed')
        def count_tokens(self, text, _): return 5000 if 'Answer 1\n' in text else 10
        def predict(self, text, questions, *, timeout_seconds):
            assert (output/'judge-operating-point.json').is_file()
            assert timeout_seconds == 60
            # Deliberately incomplete/failing cases stay in every report total.
            if 'Answer 2\n' in text: raise JudgeInputOverflow('synthetic full-prompt overflow')
            if 'Answer 3\n' in text: raise ValueError('synthetic malformed result')
            return {q['id']: 'approved' for q in questions}
    monkeypatch.setattr('rateloop_evaluator.judge_benchmark.OllamaJudge', Judge)
    # Ensure failures in both languages regardless of deterministic role hash.
    representatives = test_representatives(manifest, rows)
    counter = [0]
    def length(*_):
        counter[0] += 1
        return 5000 if counter[0] <= 2 else 10
    monkeypatch.setattr(Judge, 'count_tokens', length)
    report = run_local_judge(manifest, rows, model_dir='explicit-local-bundle', output=output)
    assert report['cohort']['source_groups'] == len(representatives)
    assert report['cohort']['states']['overflow'] >= 2
    assert sum(v['source_groups'] for v in report['cohort']['languages'].values()) == len(representatives)
    assert all(v['source_groups'] > 0 for v in report['cohort']['languages'].values())
    assert report['calibrated_confidence'] is report['qualified'] is report['activation_changed'] is False
    assert events == ['closed']
    observations = json.loads((output/'judge-observations.json').read_text())
    assert all(not {'scores', 'probabilities', 'confidence', 'input'} & item.keys() for item in observations)
    assert report['cohort']['completion_coverage'] < 1
    point = json.loads((output/'judge-operating-point.json').read_text())
    bad = deepcopy(observations); bad[0]['confidence'] = .99
    with pytest.raises(ValueError, match='numeric confidence'):
        score_label_only(manifest, rows, point, bad)
    with pytest.raises(ValueError, match='every frozen'):
        score_label_only(manifest, rows, point, observations[:-1])
    point['model']['model'] = 'different-model'
    with pytest.raises(ValueError, match='frozen model'):
        score_label_only(manifest, rows, point, observations)


def test_full_and_matched_cohorts_do_not_hide_fast_rejections_or_missing_german():
    manifest, rows = bilingual_cohort()
    first = [{'evaluation_id': row['evaluation_id'], 'state': 'completed', 'total_ms': 100,
              'benchmark_commitment': manifest['commitment']} for _, row in test_representatives(manifest, rows)]
    second = deepcopy(first)
    first[0].update(state='overflow', total_ms=1)
    second[1].update(state='failed', total_ms=2)
    report = matched_cohort_report(manifest, rows, {'base': first, 'challenger': second})
    assert report['common_completed_cases'] == len(first)-2
    assert report['full_cohort']['base']['source_groups'] == len(first)
    assert report['full_cohort']['challenger']['source_groups'] == len(first)
    assert report['full_cohort']['base']['states']['overflow'] == 1
    assert report['common_completed_latency']['base']['source_groups'] == len(first)-2
    with pytest.raises(ValueError, match='every frozen'):
        matched_cohort_report(manifest, rows, {'base': first[:-1], 'challenger': second})
    english = [r for r in rows if r['language'] == 'en']
    english_manifest = freeze_benchmark(english, SOURCES)
    observations = [{'evaluation_id': r['evaluation_id'], 'state': 'failed', 'total_ms': 1}
                    for _, r in test_representatives(english_manifest, english)]
    missing = full_cohort_report(english_manifest, english, observations)['languages']['de']
    assert missing['source_groups'] == 0 and missing['completion_coverage'] is None
