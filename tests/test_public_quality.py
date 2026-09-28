from copy import deepcopy
import gzip
import hashlib
import json
import pytest

from rateloop_evaluator.public_quality import SOURCE, prepare_cohorts, read_source


class Tokenizer:
    def count_tokens(self, text, questions):
        return 600 if 'oversized' in text else 30


def source(prefix, count=150):
    return [{'prompt': f'{prefix} prompt {i}', 'response': f'{prefix} answer {i}',
        'helpfulness': i % 5, 'correctness': (i+1) % 5} for i in range(count)]


def test_pinned_source_fails_closed_for_different_bytes(tmp_path, monkeypatch):
    path = tmp_path/'source.gz'
    path.write_bytes(gzip.compress(json.dumps({'public': True}).encode()))
    with pytest.raises(ValueError, match='pinned'):
        read_source(path, 'train')
    monkeypatch.setitem(SOURCE['files'], 'train', hashlib.sha256(path.read_bytes()).hexdigest())
    assert read_source(path, 'train') == [{'public': True}]


def test_official_final_split_never_leaks_into_development_and_middle_labels_excluded():
    train, test = source('train'), source('test')
    test.append(deepcopy(train[0]))
    test.append({'prompt': 'new source', 'response': 'oversized', 'helpfulness': 4, 'correctness': 4})
    result = prepare_cohorts(train, test, score='helpfulness', backend=Tokenizer(), max_train_groups=100, max_test_groups=100)
    assert len(result['development']) == len(result['final_test']) == 100
    assert {r['group_id'] for r in result['development']}.isdisjoint(r['group_id'] for r in result['final_test'])
    evidence = result['provenance']
    assert evidence['excluded']['final_test']['source_prompt_in_upstream_train'] == 1
    assert evidence['excluded']['final_test']['over_token_limit'] == 1
    assert evidence['excluded']['development']['middle_score_excluded'] == 30
    assert evidence['quality_gate'] is False and evidence['label_provenance'] == 'external_human'
    assert result == prepare_cohorts(list(reversed(train)), list(reversed(test)), score='helpfulness', backend=Tokenizer(), max_train_groups=100, max_test_groups=100)


def test_source_revisions_count_once_and_no_oversized_truncation():
    train, test = source('train'), source('test')
    train.extend(deepcopy(train))
    result = prepare_cohorts(train, test, score='correctness', backend=Tokenizer(), max_train_groups=100, max_test_groups=100)
    assert len({r['group_id'] for r in result['development']}) == 100
    assert result['provenance']['excluded']['development']['additional_response_same_prompt'] == 120


def test_insufficient_usable_public_examples_fail_before_claiming_representativeness():
    with pytest.raises(ValueError, match='100'):
        prepare_cohorts(source('train', 10), source('test'), score='correctness', backend=Tokenizer())


def test_reserved_tail_is_unseen_and_does_not_change_original_development_groups():
    train, test = source('train', 1500), source('test')
    original = prepare_cohorts(train, test, score='helpfulness', backend=Tokenizer(), max_train_groups=100, max_test_groups=100)
    reserved = prepare_cohorts(train, test, score='helpfulness', backend=Tokenizer(), max_train_groups=100, max_test_groups=100, reserve_unseen_tail=True)
    assert reserved['development'] == original['development']
    final_ids = {r['group_id'] for r in reserved['final_test']}
    assert final_ids.isdisjoint(r['group_id'] for r in original['development'] + original['final_test'])
    assert reserved['provenance']['final_test_upstream_split'] == 'reserved_unseen_train_tail'
