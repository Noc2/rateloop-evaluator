"""Bounded, encrypted training imports. Uploaded labels are never blind references.

This module does not perform inference, download models or execute uploaded code.
A training-only grant is sufficient; sharing and public weights remain separate.
"""
from __future__ import annotations

from copy import deepcopy
import csv
import io
import json
import time
from typing import Any, Literal

from pydantic import TypeAdapter

from .learning import LearningStore, _digest, _json
from .protocol import CaseInput, Identifier, Template, commitment

MAX_BYTES = 2 * 1024 * 1024
MAX_ROWS = 2_000
MAX_WORKSPACE_BYTES = 20 * 1024 * 1024
MAX_WORKSPACE_VERSIONS = 50
PROVENANCES = frozenset({'owner', 'ai_assisted', 'synthetic'})
_IDENTIFIER = TypeAdapter(Identifier)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON keys are not allowed')
        result[key] = value
    return result


def preview_dataset(*, template: Template | dict, content: bytes | str,
                    format: Literal['csv', 'jsonl'], mapping: dict | None = None) -> dict:
    """Validate every row before retention; callers decide whether to show a preview.

    Mapping uses text/context/evidence/case_id/group_id column names and a labels
    object keyed by criterion ID. JSONL accepts the same flat column names; nested
    objects and flags claiming human independence are not interpreted.
    """
    template = template if isinstance(template, Template) else Template.model_validate(template)
    raw = content.encode('utf-8') if isinstance(content, str) else content
    if not isinstance(raw, bytes) or not raw or len(raw) > MAX_BYTES:
        raise ValueError('Dataset must be nonempty UTF-8 and at most 2 MiB')
    text = raw.decode('utf-8-sig', errors='strict')
    if '\x00' in text:
        raise ValueError('Dataset cannot contain NUL characters')
    if format == 'csv':
        reader = csv.DictReader(io.StringIO(text, newline=''), strict=True)
        columns = reader.fieldnames
        if not columns or len(columns) != len(set(columns)) or any(not c for c in columns):
            raise ValueError('CSV requires unique nonempty column names')
        raw_rows = []
        try:
            parsed_rows = list(reader)
        except csv.Error:
            raise ValueError('CSV structure is invalid') from None
        for row in parsed_rows:
            if None in row or any(value is None for value in row.values()):
                raise ValueError('Every CSV row must match the header')
            raw_rows.append(row)
            if len(raw_rows) > MAX_ROWS:
                raise ValueError('Dataset exceeds the 2000-row limit')
    elif format == 'jsonl':
        raw_rows = []
        for line in text.splitlines():
            if not line.strip():
                continue
            row = json.loads(line, object_pairs_hook=_unique_object,
                             parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON value')))
            if not isinstance(row, dict):
                raise ValueError('Each JSONL line must be an object')
            raw_rows.append(row)
            if len(raw_rows) > MAX_ROWS:
                raise ValueError('Dataset exceeds the 2000-row limit')
    else:
        raise ValueError('Dataset format must be csv or jsonl')
    if not raw_rows:
        raise ValueError('Dataset contains no examples')
    mapping = deepcopy(mapping) if mapping is not None else {
        'text': 'text', 'context': 'context', 'evidence': 'evidence',
        'case_id': 'case_id', 'group_id': 'group_id',
        'labels': {q.id: 'label' if len(template.questions) == 1 else 'label.'+q.id for q in template.questions},
    }
    if (not isinstance(mapping, dict) or set(mapping)-{'text', 'context', 'evidence', 'case_id', 'group_id', 'labels'}
            or not isinstance(mapping.get('labels'), dict)
            or set(mapping['labels']) != {q.id for q in template.questions}):
        raise ValueError('Mapping must declare a text field and every criterion label')
    mapped_columns = [value for key, value in mapping.items() if key != 'labels'] + list(mapping['labels'].values())
    if ('text' not in mapping or 'case_id' not in mapping or 'group_id' not in mapping
            or any(not isinstance(value, str) or not value for value in mapped_columns)
            or len(mapped_columns) != len(set(mapped_columns))):
        raise ValueError('Text, case and source-group IDs require distinct named columns')
    rows, seen, counts = [], set(), {q.id: {label.id: 0 for label in q.labels} for q in template.questions}
    for index, source in enumerate(raw_rows, 1):
        try:
            case_id = _IDENTIFIER.validate_python(source[mapping['case_id']])
            group_id = _IDENTIFIER.validate_python(source[mapping['group_id']])
            if case_id in seen:
                raise ValueError('Case IDs must be unique within a dataset version')
            seen.add(case_id)
            payload = CaseInput.model_validate({key: source.get(mapping[key], '') for key in ('text', 'context', 'evidence') if key in mapping})
            if not payload.text.strip():
                raise ValueError('Material to evaluate cannot be blank')
            labels = {qid: source[column] for qid, column in mapping['labels'].items()}
            if any(not isinstance(label, str) or label not in counts[qid] for qid, label in labels.items()):
                raise ValueError('Labels must exactly match the supplied template')
            for qid, label in labels.items():
                counts[qid][label] += 1
            rows.append({'case_id': case_id, 'group_id': group_id, 'input': payload.model_dump(), 'labels': labels})
        except (KeyError, TypeError, ValueError) as error:
            # Never echo user material or the offending label into server logs.
            raise ValueError(f'Dataset row {index} does not match the declared fields, IDs or labels') from None
    return {'template': template.model_dump(), 'rows': rows, 'row_count': len(rows),
            'group_count': len({row['group_id'] for row in rows}), 'label_counts': counts,
            'token_validation': 'Exact tokenizer limits are checked before inference or training; no truncation is performed.'}


def import_dataset(store: LearningStore, *, workspace_id: str, dataset_id: str,
                   template: Template | dict, content: bytes | str, format: Literal['csv', 'jsonl'],
                   provenance: str, evidence: str, mapping: dict | None = None,
                   model_bundle_id: str | None = None, now: float | None = None) -> dict:
    """Atomically import an immutable version under private-training permission.

    Labels keep their declared source. Only the separate authenticated blind
    review path can produce independent references used in qualification gates.
    """
    _IDENTIFIER.validate_python(workspace_id)
    _IDENTIFIER.validate_python(dataset_id)
    if provenance not in PROVENANCES or not isinstance(evidence, str) or not 1 <= len(evidence.strip()) <= 1000:
        raise ValueError('Declare owner, AI-assisted or synthetic label provenance and authorization evidence')
    preview = preview_dataset(template=template, content=content, format=format, mapping=mapping)
    current = time.time() if now is None else now
    committed_template = commitment(preview['template'], 'rateloop.evaluator.template.v1')
    identity = {'workspace_id': workspace_id, 'dataset_id': dataset_id, 'template': preview['template'],
                'rows': preview['rows'], 'provenance': provenance, 'evidence': evidence,
                'model_bundle_id': model_bundle_id}
    version_id = 'dataset_'+_digest(identity)
    with store.transaction() as state:
        versions = state.setdefault('datasets', {})
        examples = state.setdefault('dataset_examples', {})
        grant_ids = set()
        for row in preview['rows']:
            fields = ['input.'+key for key, value in row['input'].items() if value]
            grant_ids.update(store._matching_grants(state, workspace_id=workspace_id, right='private_training',
                case_id=row['case_id'], template_id=preview['template']['id'], fields=[*fields, 'imported_labels'],
                model_bundle_id=model_bundle_id, template_commitment=committed_template, now=current))
        if version_id in versions:
            _validate_version(store, state, versions[version_id], workspace_id, current)
            return deepcopy(versions[version_id])
        workspace_versions = [v for v in versions.values() if v['workspace_id'] == workspace_id]
        stored_bytes = sum(len(_json(row)) for row in examples.values() if row['workspace_id'] == workspace_id)
        if len(workspace_versions) >= MAX_WORKSPACE_VERSIONS or stored_bytes+len(_json(identity)) > MAX_WORKSPACE_BYTES:
            raise ValueError('Workspace dataset capacity exceeded')
        example_ids = []
        for row in preview['rows']:
            example_id = 'sample_'+_digest([version_id, row['case_id']])
            example_ids.append(example_id)
            fields = ['input.'+key for key, value in row['input'].items() if value]
            examples[example_id] = {'id': example_id, 'dataset_version_id': version_id,
                'source_kind': 'dataset', 'workspace_id': workspace_id, **deepcopy(row),
                'template': deepcopy(preview['template']), 'template_commitment': committed_template,
                'input_commitment': commitment(row['input'], 'rateloop.dataset.input.v1'),
                'model_bundle_id': model_bundle_id, 'fields': fields,
                'label_provenance': provenance, 'independent_reference': False,
                'training_eligible': True, 'created_at': current, 'grant_ids': sorted(grant_ids)}
        version = {'id': version_id, 'dataset_id': dataset_id, 'workspace_id': workspace_id,
            'version': 1+max((v['version'] for v in workspace_versions if v['dataset_id'] == dataset_id), default=0),
            'template_commitment': committed_template, 'template_id': preview['template']['id'],
            'template_version': preview['template']['version'], 'provenance': provenance,
            'independent_reference': False, 'training_eligible': True, 'evidence': evidence,
            'row_count': preview['row_count'], 'group_count': preview['group_count'],
            'label_counts': preview['label_counts'], 'example_ids': example_ids,
            'grant_ids': sorted(grant_ids), 'created_at': current, 'invalidated_at': None}
        versions[version_id] = version
        return deepcopy(version)


def _validate_version(store, state, version, workspace_id, now):
    if version['workspace_id'] != workspace_id:
        raise KeyError('Dataset version not found')
    if version.get('invalidated_at') is not None:
        raise PermissionError('Dataset version has been invalidated')
    for grant_id in version['grant_ids']:
        grant = state['grants'].get(grant_id)
        if (not grant or grant['workspace_id'] != workspace_id or grant['revoked_at'] is not None
                or grant['expires_at'] <= now
                or grant.get('authorization_until') is not None and grant['authorization_until'] <= now):
            raise PermissionError('Dataset source permission expired or was revoked')
    for example_id in version['example_ids']:
        row = state['dataset_examples'][example_id]
        store._matching_grants(state, workspace_id=workspace_id, right='private_training',
            case_id=row['case_id'], template_id=row['template']['id'], fields=[*row['fields'], 'imported_labels'],
            model_bundle_id=row.get('model_bundle_id'), template_commitment=row['template_commitment'], now=now)


def load_dataset(store: LearningStore, version_id: str, workspace_id: str, *, now: float | None = None) -> dict:
    with store.transaction() as state:
        version = state.get('datasets', {}).get(version_id)
        if not version:
            raise KeyError('Dataset version not found')
        _validate_version(store, state, version, workspace_id, time.time() if now is None else now)
        return {**deepcopy(version), 'rows': [deepcopy(state['dataset_examples'][key]) for key in version['example_ids']]}
