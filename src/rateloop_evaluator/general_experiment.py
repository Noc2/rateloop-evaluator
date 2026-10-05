"""Runnable offline public benchmark preparation and pinned-model measurement."""
from __future__ import annotations

from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import platform
import time

from .backends import GLiNERBackend, GLiClassBackend, offline_environment, render_input, tokenizer_commitment
from .calibration import apply_temperature, fit_temperature
from .general_benchmark import digest, freeze_benchmark, verify_manifest
from .general_qualification import freeze_operating_point, score_general_benchmark, test_representatives
from .public_quality import SOURCE, read_source, _normalized, _hash
from .templates import custom_text_evaluation


def write_private(path, value):
    """Exclusive writes preserve predeclared plans and earlier test observations."""
    import os
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, 'w') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')


def prepare_public(source_path, *, criterion='helpfulness', maximum_groups=600):
    if criterion not in ('helpfulness', 'correctness') or type(maximum_groups) is not int or not 100 <= maximum_groups <= 1200:
        raise ValueError('Choose helpfulness/correctness and 100–1,200 source groups')
    source = read_source(source_path, 'train')
    question = {'helpfulness': 'Does the response helpfully fulfil the request in context, with no major omission or irrelevant answer?',
                'correctness': 'Is the response factually and logically correct for the request in context, without a major error?'}[criterion]
    template = custom_text_evaluation('en', question, 'Meets the criterion', 'Does not meet the criterion').model_dump()
    groups = {}; exclusions = Counter()
    for raw in sorted(source, key=lambda r: _hash(r['prompt'] + '\0' + r['response'])):
        value = raw[criterion]
        if type(value) is not int or not 0 <= value <= 4:
            raise ValueError('Unexpected pinned human source score')
        if value == 2:
            exclusions['middle_score'] += 1; continue
        group = 'prompt_' + _hash(_normalized(raw['prompt']))
        if group in groups:
            exclusions['additional_response_same_prompt'] += 1; continue
        if not raw['response'].strip():
            exclusions['empty_response'] += 1; continue
        identity = _hash(raw['prompt'] + '\0' + raw['response'])
        groups[group] = {'evaluation_id': 'hs_' + identity, 'group_id': group,
            'family': 'request_following' if criterion == 'helpfulness' else 'knowledge',
            'language': 'en', 'source': 'helpsteer2',
            'input': {'text': raw['response'], 'context': raw['prompt'], 'evidence': ''},
            'template': template, 'labels': {'judgment': 'approved' if value >= 3 else 'rejected'}}
    rows = sorted(groups.values(), key=lambda r: _hash('rateloop.general-public.v1:' + r['group_id']))[:maximum_groups]
    exclusions['beyond_prespecified_group_limit'] = len(groups) - len(rows)
    sources = {'helpsteer2': {k: SOURCE[k] for k in ('revision', 'license', 'attribution', 'url', 'label_provenance')}}
    sources['helpsteer2'].update({'sha256': SOURCE['files']['train'], 'criterion': criterion,
        'positive_scores': [3, 4], 'negative_scores': [0, 1], 'excluded': dict(exclusions),
        'limits': 'English public human-score extremes. Family is a declared rubric scope, not an independently labeled task taxonomy. '
                  'No German/source-faithfulness coverage. Base pretraining contamination is unknown. Long inputs remain in coverage.'})
    return rows, sources


def _predict(backend, row):
    started = time.perf_counter()
    try:
        text = render_input(row['input'])
        if backend.count_tokens(text, row['template']['questions']) > row['template']['maxTokens']:
            return {'state': 'overflow', 'total_ms': (time.perf_counter() - started) * 1000, 'cost_usd': 0, 'queue_ms': 0}
        scores = backend.predict(text, row['template']['questions'])
        # Validate before persisting to avoid reporting malformed scores as success.
        from .quality import score_predictions
        score_predictions([row], [scores])
        return {'state': 'completed', 'scores': scores, 'total_ms': (time.perf_counter() - started) * 1000,
                'queue_ms': 0, 'cost_usd': 0}
    except (ValueError, RuntimeError):
        # No raw content or exception strings in operational reports.
        return {'state': 'failed', 'total_ms': (time.perf_counter() - started) * 1000, 'queue_ms': 0, 'cost_usd': 0}


def run_local(manifest, rows, *, model_dir, output, backend_name='gliner', device='cpu', threshold=.9):
    """Fit diagnostic temperatures on calibration groups, then score test once.

    Public inputs are authorized by this explicit offline command. The runner has
    no network client, service credentials, production registration or fallback.
    """
    verify_manifest(manifest, rows)
    if backend_name not in ('gliner', 'gliclass'):
        raise ValueError('Unknown pinned local adapter')
    output = Path(output).resolve(); output.mkdir(mode=0o700, parents=True, exist_ok=False)
    offline_environment()
    backend = (GLiNERBackend if backend_name == 'gliner' else GLiClassBackend)(model_dir, device)
    started = time.perf_counter(); backend.load()
    load_ms = (time.perf_counter() - started) * 1000
    weights = backend.manifest['files']['model.safetensors']
    library = 'gliner2' if backend_name == 'gliner' else 'gliclass'
    model = {'weights_sha256': weights, 'tokenizer_sha256': tokenizer_commitment(backend.manifest).removeprefix('sha256:'),
             'adapter': backend_name, 'runtime': {'python': platform.python_version(), library: importlib.metadata.version(library),
                 'torch': importlib.metadata.version('torch'), 'device': device},
             'quantization': 'float32', 'artifact_manifest_commitment': digest(backend.manifest)}
    route = {'id': 'local-full-input', 'version': 1, 'evidence_policy': 'supplied-input-only; no lookup; no truncation'}
    raw_point = freeze_operating_point(manifest, model=model, route=route, threshold=threshold)
    write_private(output/'raw-operating-point.json', raw_point)
    by_id = {row['evaluation_id']: row for row in rows}; calibration_groups = {}
    for entry in manifest['entries']:
        if entry['role'] == 'calibration':
            calibration_groups.setdefault((entry['template_commitment'], entry['group']), (entry, by_id[entry['evaluation_id']]))
    successful = {}; calibration_states = Counter()
    for entry, row in calibration_groups.values():
        observation = _predict(backend, row); calibration_states[observation['state']] += 1
        if observation['state'] == 'completed':
            successful.setdefault(entry['template_commitment'], []).append((entry, row, observation['scores']))
    calibrations = {}
    for scope, cases in successful.items():
        for question in cases[0][1]['template']['questions']:
            qid = question['id']
            # Sparse/one-class data cannot give useful diagnostic calibration.
            if len(cases) < 20 or len({r['labels'][qid] for _, r, _ in cases}) < 2:
                continue
            artifact = fit_temperature([s[qid] for _, _, s in cases], [r['labels'][qid] for _, r, _ in cases],
                model_bundle_id='public-general-diagnostic', template_commitment=scope, question_id=qid,
                language=cases[0][1]['language'], example_ids=[e['group'] for e, _, _ in cases], model_weights_sha256=weights)
            calibrations[scope + '/' + qid] = artifact
    write_private(output/'diagnostic-calibrations.json', calibrations)
    calibrated_point = freeze_operating_point(manifest, model=model, route=route, threshold=threshold,
        calibration_commitment=digest(calibrations)) if calibrations else None
    if calibrated_point:
        write_private(output/'calibrated-operating-point.json', calibrated_point)
    # Both immutable points/calibrators are on disk before observing final scores.
    raw = []; calibrated = []
    for entry, row in test_representatives(manifest, rows):
        observation = _predict(backend, row)
        raw.append({**observation, 'evaluation_id': row['evaluation_id'], 'operating_point_commitment': raw_point['commitment']})
        if calibrated_point:
            transformed = dict(observation)
            scope = entry['template_commitment']
            if observation['state'] == 'completed':
                scores = {}
                for question in row['template']['questions']:
                    qid = question['id']; artifact = calibrations.get(scope + '/' + qid)
                    if not artifact:
                        transformed = {k: v for k, v in observation.items() if k != 'scores'}
                        transformed['state'] = 'not_checked'
                        break
                    scores[qid] = apply_temperature(observation['scores'][qid], artifact,
                        model_bundle_id='public-general-diagnostic', template_commitment=scope, question_id=qid, language=row['language'])
                else:
                    transformed['scores'] = scores
            calibrated.append({**transformed, 'evaluation_id': row['evaluation_id'],
                               'operating_point_commitment': calibrated_point['commitment']})
    write_private(output/'raw-observations.json', raw)
    raw_report = score_general_benchmark(manifest, rows, raw_point, raw)
    report = {'schema_version': 'rateloop.general-local-experiment.v1', 'raw': raw_report,
        'calibrated': score_general_benchmark(manifest, rows, calibrated_point, calibrated) if calibrated_point else None,
        'calibration_states': dict(calibration_states), 'model_load_ms': load_ms, 'model': model,
        'measurement_limits': 'Single local process, warm-model input validation/tokenization/inference latency; '
            'model load reported separately. No website/network/hosted queue or concurrency SLA. '
            'API cost zero; hardware/electricity costs are not measured. External public labels only.',
        'qualified': False, 'activation_changed': False}
    if calibrated_point:
        write_private(output/'calibrated-observations.json', calibrated)
    write_private(output/'report.json', report)
    return report
