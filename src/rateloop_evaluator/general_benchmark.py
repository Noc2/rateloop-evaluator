"""Frozen general-request diagnostic cohorts. Imports never confer qualification.

Files contain public or explicitly authorized inputs and stay outside source control.
The exported manifest contains commitments/metadata only. Related sources share a
role; a later relationship crossing frozen roles fails instead of moving test data.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re
import unicodedata

from .learning import _source_aliases

FAMILIES = ('request_following', 'source_faithfulness', 'knowledge', 'writing', 'code_math', 'unsupported')
LANGUAGES = ('en', 'de')
ROLES = ('development', 'calibration', 'test')
PROVENANCE = ('external_human', 'owner', 'ai_assisted', 'synthetic')
SCHEMA = 'rateloop.general-benchmark.v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _text(value, name, maximum=256):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f'{name} must be nonempty bounded text')
    return value


def _normalized(value):
    return ' '.join(unicodedata.normalize('NFKC', value).casefold().split())


def _aliases(row):
    aliases = list(_source_aliases({**row, 'case_id': row['evaluation_id']}))
    # Prompt/document identities keep all answers/revisions together. Explicit
    # related_source_ids are required for semantic paraphrases/translated pairs.
    for field in ('context', 'evidence'):
        if row['input'].get(field, '').strip():
            aliases.append(field + ':' + digest(_normalized(row['input'][field])))
    aliases.extend('related:' + _text(value, 'Related source ID') for value in row.get('related_source_ids', []))
    return sorted({digest(value) for value in aliases})


def validate_rows(rows, sources):
    if not isinstance(rows, list) or not 1 <= len(rows) <= 20000:
        raise ValueError('Supply between one and 20,000 benchmark rows')
    if not isinstance(sources, dict) or not sources:
        raise ValueError('Source attribution is required')
    for source in sources.values():
        for key in ('revision', 'license', 'attribution', 'url'):
            _text(source.get(key), f'Source {key}', 2048)
        if source.get('label_provenance') not in PROVENANCE:
            raise ValueError('An import cannot assert authenticated independent human status')
        if not re.fullmatch('[0-9a-f]{64}', source.get('sha256', '')):
            raise ValueError('Sources must bind explicitly obtained bytes by SHA-256')
    seen = set()
    for row in rows:
        identity = _text(row.get('evaluation_id'), 'Evaluation ID')
        if identity in seen:
            raise ValueError('Duplicate evaluation ID')
        seen.add(identity)
        _text(row.get('group_id'), 'Source group')
        if row.get('family') not in FAMILIES or row.get('language') not in LANGUAGES:
            raise ValueError('Unknown diagnostic task family or language')
        if row.get('source') not in sources:
            raise ValueError('Row has no attributed source')
        if not isinstance(row.get('related_source_ids', []), list):
            raise ValueError('Related source IDs must be a list')
        payload = row.get('input')
        if not isinstance(payload, dict) or set(payload) - {'text', 'context', 'evidence'}:
            raise ValueError('Input must contain only text, context and evidence')
        _text(payload.get('text'), 'Answer', 200000)
        for value in payload.values():
            if not isinstance(value, str) or len(value) > 200000:
                raise ValueError('Input fields must be bounded text; do not silently truncate')
        template = row.get('template', {})
        if template.get('language') != row['language'] or not template.get('questions'):
            raise ValueError('Template must declare the exact language and questions')
        expected = {}
        for question in template['questions']:
            qid = _text(question.get('id'), 'Question ID')
            labels = [label['id'] for label in question.get('labels', [])]
            if qid in expected or not 2 <= len(labels) <= 10 or len(set(labels)) != len(labels):
                raise ValueError('Questions need unique IDs and two to ten distinct labels')
            for label in labels:
                _text(label, 'Label ID')
            if not set(question.get('passLabels', [])) <= set(labels):
                raise ValueError('Passing labels must belong to the rubric')
            expected[qid] = labels
        if set(row.get('labels', {})) != set(expected) or any(row['labels'][q] not in labels for q, labels in expected.items()):
            raise ValueError('Each criterion needs a declared reference label')


def freeze_benchmark(rows, sources, *, previous=None, sampling='diagnostic_balanced'):
    """Freeze deterministic 50/25/25 roles and alias assignments across versions.

    Exact counts vary: labels never influence assignment and sparse slices stay
    sparse. No forced rebalancing moves previously held-out groups into training.
    """
    validate_rows(rows, sources)
    if sampling not in ('diagnostic_balanced', 'representative_traffic'):
        raise ValueError('Declare diagnostic or representative sampling')
    if previous:
        verify_manifest(previous)
        if previous['sampling'] != sampling:
            raise ValueError('Sampling design cannot change within a frozen benchmark')
    ledger = deepcopy(previous['alias_assignments']) if previous else {}
    aliases = {row['evaluation_id']: _aliases(row) for row in rows}
    # Connected components catch relationships within the same import before any
    # role is assigned, avoiding row-order dependent leakage.
    parent = {}
    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    def join(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)
    for values in aliases.values():
        for value in values:
            join(values[0], value)
    components = {}
    for value in parent:
        components.setdefault(find(value), []).append(value)
    assignments = {}
    for component, values in components.items():
        known = {ledger[a]['group'] for a in values if a in ledger}
        roles = {ledger[a]['role'] for a in values if a in ledger}
        if len(roles) > 1:
            raise ValueError('Related sources bridge frozen learning partitions')
        group = min(known) if known else 'source_' + min(values)
        fraction = int(digest(['rateloop.general-split.v1', group]), 16) / 2**256
        role = next(iter(roles)) if roles else ('development' if fraction < .5 else 'calibration' if fraction < .75 else 'test')
        assignment = {'group': group, 'role': role}
        assignments[component] = assignment
        # A new relationship in one partition unifies previously counted groups.
        for alias, old in ledger.items():
            if old['group'] in known:
                ledger[alias] = assignment
        ledger.update({value: assignment for value in values})
    entries = []
    for row in sorted(rows, key=lambda r: r['evaluation_id']):
        assignment = assignments[find(aliases[row['evaluation_id']][0])]
        entries.append({'evaluation_id': row['evaluation_id'], **assignment,
            'family': row['family'], 'language': row['language'], 'source': row['source'],
            'input_commitment': digest(row['input']), 'template_commitment': digest(row['template']),
            'labels_commitment': digest(row['labels'])})
    manifest = {'schema_version': SCHEMA, 'sampling': sampling,
        'sources': deepcopy(sources), 'entries': entries, 'alias_assignments': ledger,
        'previous_commitment': previous['commitment'] if previous else None,
        'split_policy': {'development': .5, 'calibration': .25, 'test': .25,
                         'development_validation': 'Reserve existing fixed 20% validation split before fitting.'},
        'coverage': {f'{family}/{language}': {
            role: len({e['group'] for e in entries if e['family'] == family and e['language'] == language and e['role'] == role})
            for role in ROLES} for family in FAMILIES for language in LANGUAGES},
        'quality_gate': False, 'independent_reference_count': 0}
    manifest['commitment'] = digest(manifest)
    return manifest


def verify_manifest(manifest, rows=None):
    if not isinstance(manifest, dict) or manifest.get('schema_version') != SCHEMA:
        raise ValueError('Unsupported general benchmark manifest')
    value = {key: item for key, item in manifest.items() if key != 'commitment'}
    if digest(value) != manifest.get('commitment') or manifest.get('quality_gate') is not False:
        raise ValueError('Benchmark commitment or diagnostic status differs')
    if rows is not None:
        validate_rows(rows, manifest['sources'])
        expected = {e['evaluation_id']: e for e in manifest['entries']}
        if set(expected) != {row['evaluation_id'] for row in rows} or len(expected) != len(rows):
            raise ValueError('Benchmark data must cover every frozen row exactly once')
        for row in rows:
            entry = expected[row['evaluation_id']]
            if any(manifest['alias_assignments'].get(alias) != {'group': entry['group'], 'role': entry['role']}
                   for alias in _aliases(row)):
                raise ValueError('Source grouping changed after freezing')
            for field in ('input', 'template', 'labels'):
                if digest(row[field]) != entry[field + '_commitment']:
                    raise ValueError('Input, rubric or references changed after freezing')
            if any(row[field] != entry[field] for field in ('family', 'language', 'source')):
                raise ValueError('Benchmark scope changed after freezing')


def blind_review_packet(manifest, rows):
    """Return cases and anchored rubrics, never AI predictions/imported labels.

    This prepares an independent collection; it does not authenticate a reviewer
    or transform external labels into trusted feedback. Use the review service.
    """
    verify_manifest(manifest, rows)
    return {'schema_version': 'rateloop.blind-benchmark-review.v1',
        'benchmark_commitment': manifest['commitment'],
        'instructions': [
            'Two eligible reviewers work independently without seeing AI ratings or imported labels.',
            'Judge only each stated criterion against the request, answer and supplied evidence.',
            'Use insufficient evidence when information is missing; do not infer factual truth from fluency.',
            'Record short evidence for the label. Preserve disagreement for a separate blinded adjudicator.',
            'Submit authenticated feedback through the review service; this file grants no reviewer identity.'],
        'cases': [{'evaluation_id': row['evaluation_id'], 'language': row['language'],
                   'input': deepcopy(row['input']), 'questions': deepcopy(row['template']['questions'])}
                  for row in sorted(rows, key=lambda r: r['evaluation_id'])]}
