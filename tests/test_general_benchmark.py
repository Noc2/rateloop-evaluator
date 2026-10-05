from copy import deepcopy
import pytest

from rateloop_evaluator.general_benchmark import (
    blind_review_packet, digest, freeze_benchmark, verify_manifest,
)

SOURCES = {'public': {'revision': 'abc123', 'license': 'CC-BY-4.0',
    'attribution': 'Public author', 'url': 'https://example.org/data',
    'sha256': 'a' * 64, 'label_provenance': 'external_human'}}


def row(i, **kwargs):
    value = {'evaluation_id': f'case-{i}', 'group_id': f'group-{i}',
        'family': 'request_following', 'language': 'en', 'source': 'public',
        'input': {'text': f'Answer {i}', 'context': f'Prompt {i}', 'evidence': ''},
        'template': {'language': 'en', 'questions': [{'id': 'judgment', 'prompt': 'Follows request?',
            'labels': [{'id': 'approved'}, {'id': 'rejected'}], 'passLabels': ['approved']}]},
        'labels': {'judgment': 'approved'}}
    value.update(kwargs)
    return value


def test_freeze_preserves_splits_labels_do_not_drive_roles_and_no_raw_text():
    rows = [row(i) for i in range(100)]
    frozen = freeze_benchmark(rows, SOURCES)
    assert all(frozen['coverage']['request_following/en'][part] for part in ('development', 'calibration', 'test'))
    assert frozen['coverage']['source_faithfulness/de'] == dict.fromkeys(('development', 'calibration', 'test'), 0)
    assert frozen['quality_gate'] is False
    assert frozen['independent_reference_count'] == 0
    assert 'Answer' not in str(frozen)
    flipped = [{**r, 'labels': {'judgment': 'rejected'}} for r in rows]
    second = freeze_benchmark(flipped + [row(101)], SOURCES, previous=frozen)
    assert [(e['group'], e['role']) for e in frozen['entries']] == [
        (e['group'], e['role']) for e in second['entries'] if e['evaluation_id'] != 'case-101']
    verify_manifest(frozen, rows)
    with pytest.raises(ValueError, match='references changed'):
        verify_manifest(frozen, flipped)


def test_related_translation_prompt_and_document_sources_stay_together():
    rows = [row(0), row(1), row(2), row(3)]
    rows[0]['related_source_ids'] = ['translated-document-17']
    rows[1]['related_source_ids'] = ['translated-document-17']
    rows[1]['language'] = rows[1]['template']['language'] = 'de'
    rows[2]['input']['context'] = '  PROMPT   0  '
    rows[2]['input']['evidence'] = 'shared source document'
    rows[3]['input']['evidence'] = 'Shared   Source Document'
    a = freeze_benchmark(rows, SOURCES)
    b = freeze_benchmark(list(reversed(rows)), SOURCES)
    assert a == b
    assert len({e['group'] for e in a['entries']}) == 1


def test_new_relationship_cannot_bridge_frozen_test_and_development():
    rows = [row(i) for i in range(100)]
    frozen = freeze_benchmark(rows, SOURCES)
    ids = {role: next(e['evaluation_id'] for e in frozen['entries'] if e['role'] == role)
           for role in ('test', 'development')}
    for r in rows:
        if r['evaluation_id'] in ids.values():
            r['related_source_ids'] = ['late-discovered-translation']
    with pytest.raises(ValueError, match='bridge frozen'):
        freeze_benchmark(rows, SOURCES, previous=frozen)


def test_blind_packet_excludes_reference_labels_and_all_predictions():
    rows = [row(1)]
    rows[0]['prediction'] = {'approved': .9}
    packet = blind_review_packet(freeze_benchmark(rows, SOURCES), rows)
    assert 'labels' not in packet['cases'][0]
    assert 'prediction' not in str(packet)
    assert packet['cases'][0]['questions'][0]['labels']
    assert len(packet['instructions']) == 5


def test_untrusted_import_cannot_claim_authenticated_reviews():
    sources = deepcopy(SOURCES)
    sources['public']['label_provenance'] = 'independent_human'
    with pytest.raises(ValueError, match='cannot assert'):
        freeze_benchmark([row(0)], sources)
    broken = freeze_benchmark([row(0)], SOURCES)
    broken['quality_gate'] = True
    broken['commitment'] = digest({k: v for k, v in broken.items() if k != 'commitment'})
    with pytest.raises(ValueError, match='diagnostic'):
        verify_manifest(broken)


def test_changed_group_or_alias_cannot_sneak_past_frozen_input_check():
    rows = [row(0)]
    frozen = freeze_benchmark(rows, SOURCES)
    rows[0]['group_id'] = 'new-group'
    with pytest.raises(ValueError, match='Source grouping'):
        verify_manifest(frozen, rows)
