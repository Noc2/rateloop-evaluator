"""Pinned HelpSteer2 benchmark preparation; public text stays outside Git.

The upstream validation split is a final holdout here. It never enters training,
checkpoint selection or calibration. This measures two declared English rubrics,
not general evaluator competence or customer-specific qualification.
"""
from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import unicodedata

from .backends import render_input
from .templates import custom_text_evaluation

SOURCE = {'repository': 'nvidia/HelpSteer2',
    'revision': '990b2711a36180dd19d9c94b8627844866f8982a',
    'license': 'CC-BY-4.0', 'attribution': 'NVIDIA HelpSteer2',
    'url': 'https://huggingface.co/datasets/nvidia/HelpSteer2',
    'label_provenance': 'external_human',
    'files': {'train': 'c0d7e91d738d42e8a08070db26c4c09a9c7631308e1f0fd380ff43d130c9f713',
              'validation': '610eeb5289494d613c4c0f70aade2df8df0b499f3a24e76d232f74e6909d010a'}}


def _hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _normalized(value):
    return ' '.join(unicodedata.normalize('NFKC', value).casefold().split())


def read_source(path: str | Path, split: str) -> list[dict]:
    """Read only explicitly downloaded pinned bytes, with no network access."""
    if split not in SOURCE['files']:
        raise ValueError('Unknown source split')
    raw = Path(path).read_bytes()
    if len(raw) > 32 * 1024 * 1024 or hashlib.sha256(raw).hexdigest() != SOURCE['files'][split]:
        raise ValueError('Public benchmark source does not match the pinned file')
    with gzip.GzipFile(fileobj=__import__('io').BytesIO(raw)) as stream:
        content = stream.read(64 * 1024 * 1024 + 1)
    if len(content) > 64 * 1024 * 1024:
        raise ValueError('Public benchmark source exceeds the extraction bound')
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def prepare_cohorts(train: list[dict], final_test: list[dict], *, score: str,
                    backend, max_train_groups: int = 600, max_test_groups: int = 200,
                    reserve_unseen_tail: bool = False) -> dict:
    if score not in {'helpfulness', 'correctness'} or not 100 <= max_train_groups <= 1500 or not 100 <= max_test_groups <= 500:
        raise ValueError('Choose helpfulness/correctness and bounded representative group counts')
    prompt = {'helpfulness': 'Does the response helpfully fulfil the request in context, with no major omission or irrelevant answer?',
              'correctness': 'Is the response factually and logically correct for the request in context, without a major error?'}[score]
    template = custom_text_evaluation('en', prompt, 'Meets the criterion', 'Does not meet the criterion').model_dump()
    if reserve_unseen_tail:
        def in_tail(row):
            group = 'prompt_' + _hash(_normalized(row['prompt']))
            return int(_hash('rateloop.public-quality.v1:' + group), 16) / 2**256 >= .8
        # Original development used ascending group hashes. The experiment
        # driver additionally proves its original development commitment is
        # unchanged, so these high-hash groups were never training/validation.
        final_test = [row for row in train if in_tail(row)]
        train = [row for row in train if not in_tail(row)]
    all_train_groups = {_hash(_normalized(row['prompt'])) for row in train}
    cohorts, exclusions = {}, {}
    for part, raw_rows, maximum in [('development', train, max_train_groups), ('final_test', final_test, max_test_groups)]:
        counts = Counter(); groups = {}
        # Ordering and representative selection precede observing predictions.
        for raw in sorted(raw_rows, key=lambda r: _hash(r['prompt'] + '\0' + r['response'])):
            group = _hash(_normalized(raw['prompt']))
            if part == 'final_test' and group in all_train_groups:
                counts['source_prompt_in_upstream_train'] += 1; continue
            value = raw[score]
            if type(value) is not int or not 0 <= value <= 4:
                raise ValueError('Unexpected source reference score')
            if value == 2:
                counts['middle_score_excluded'] += 1; continue
            payload = {'text': raw['response'], 'context': raw['prompt'], 'evidence': ''}
            if not payload['text'].strip():
                counts['empty_response'] += 1; continue
            if backend.count_tokens(render_input(payload), template['questions']) > template['maxTokens']:
                counts['over_token_limit'] += 1; continue
            if group in groups:
                counts['additional_response_same_prompt'] += 1; continue
            identity = _hash(raw['prompt'] + '\0' + raw['response'])
            groups[group] = {'evaluation_id': 'hs_' + identity, 'case_id': 'hs_' + identity,
                'group_id': 'prompt_' + group, 'input': payload,
                'labels': {'judgment': 'approved' if value >= 3 else 'rejected'},
                'template': template, 'label_provenance': 'external_human', 'independent_reference': False}
        selected = sorted(groups.values(), key=lambda r: _hash('rateloop.public-quality.v1:' + r['group_id']))[:maximum]
        counts['beyond_prespecified_group_limit'] = len(groups)-len(selected)
        if len(selected) < 100:
            raise ValueError('Fewer than 100 usable source groups; insufficient benchmark coverage')
        cohorts[part] = selected; exclusions[part] = dict(counts)
    # Avoid identical answers crossing splits even when prompt wording differs.
    development_inputs = {_hash(_normalized(row['input']['text'])) for row in cohorts['development']}
    before = len(cohorts['final_test'])
    cohorts['final_test'] = [row for row in cohorts['final_test'] if _hash(_normalized(row['input']['text'])) not in development_inputs]
    exclusions['final_test']['normalized_answer_overlap'] = before-len(cohorts['final_test'])
    if len(cohorts['final_test']) < 100:
        raise ValueError('Fewer than 100 final groups after duplicate-answer exclusions')
    return {'template': template, **cohorts, 'provenance': {**SOURCE,
        'criterion': score, 'positive_scores': [3, 4], 'negative_scores': [0, 1],
        'selection': 'One eligible response per normalized prompt; deterministic hash ordering; no prediction-based selection.',
        'final_test_upstream_split': 'reserved_unseen_train_tail' if reserve_unseen_tail else 'validation',
        'reserved_tail_fraction': .2 if reserve_unseen_tail else None, 'excluded': exclusions,
        'counts': {part: {'groups': len(rows), 'labels': dict(Counter(r['labels']['judgment'] for r in rows))}
                   for part, rows in cohorts.items()},
        'quality_gate': False,
        'limits': 'English public data, extremes of human score scale, short inputs only. External labels are not authenticated RateLoop blind reviews. Source contamination in base pretraining is unknown.'}}
