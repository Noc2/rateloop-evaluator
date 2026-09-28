"""Offline bounded experiment on explicitly downloaded public HelpSteer2 files.

No candidate is registered in the website, activated, uploaded or qualified.
Run once per new, empty output directory. Reports exclude raw public text.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

from rateloop_evaluator.backends import GLiNERBackend, offline_environment, render_input
from rateloop_evaluator.datasets import import_dataset
from rateloop_evaluator.calibration import apply_temperature, fit_temperature
from rateloop_evaluator.learning import LearningStore, provision_key
from rateloop_evaluator.public_quality import prepare_cohorts, read_source
from rateloop_evaluator.protocol import commitment
from rateloop_evaluator.quality import group_representatives, score_predictions
from rateloop_evaluator.training import TrainOptions, ValidationQualityError, train_snapshot


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    path.chmod(0o600)


def unload(backend, device):
    backend._model = None
    gc.collect()
    import torch
    if device == 'mps': torch.mps.empty_cache()
    if device == 'cuda': torch.cuda.empty_cache()


def measure(backend, rows, calibration_rows):
    started = time.perf_counter()
    calibration_rows = group_representatives(calibration_rows)
    template_hash = commitment(rows[0]['template'], 'rateloop.evaluator.template.v1')
    calibration_scores = [backend.predict(render_input(row['input']), row['template']['questions'])['judgment'] for row in calibration_rows]
    calibration = fit_temperature(calibration_scores, [r['labels']['judgment'] for r in calibration_rows],
        model_bundle_id='public-benchmark', template_commitment=template_hash, question_id='judgment', language='en',
        example_ids=[r['group_id'] for r in calibration_rows], model_weights_sha256=backend.manifest['files']['model.safetensors'])
    predictions = [backend.predict(render_input(row['input']), row['template']['questions']) for row in rows]
    report = score_predictions(rows, predictions)
    calibrated = [{'judgment': apply_temperature(scores['judgment'], calibration,
        model_bundle_id='public-benchmark', template_commitment=template_hash, question_id='judgment', language='en')} for scores in predictions]
    calibrated_report = score_predictions(rows, calibrated)
    calibrated_report['scores_calibrated'] = True
    report['calibration'] = {'temperature': calibration['temperature'], 'source_groups': len(calibration_rows),
        'partition': 'development calibration, excluded from optimization and validation selection',
        'test_metrics': calibrated_report, 'qualified': False}
    report['prediction_seconds'] = time.perf_counter()-started
    report['model_manifest'] = backend.manifest
    return report


def run(args):
    output = Path(args.output).resolve()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    offline_environment()
    train, test = read_source(args.train, 'train'), read_source(args.test, 'validation')
    backend = GLiNERBackend(args.model_dir, args.device)
    cohort = prepare_cohorts(train, test, score=args.criterion, backend=backend,
        max_train_groups=args.development_groups, max_test_groups=args.test_groups,
        reserve_unseen_tail=bool(args.previous_experiment))
    write(output/'source.json', cohort['provenance'])
    # Freeze IDs before any model score or optimizer update is observed.
    commitments = {part: hashlib.sha256(json.dumps([
        {key: row[key] for key in ('evaluation_id', 'group_id', 'labels')} for row in cohort[part]],
        sort_keys=True).encode()).hexdigest() for part in ('development', 'final_test')}
    if args.previous_experiment:
        previous = json.loads((Path(args.previous_experiment)/'cohort-commitments.json').read_text())
        if previous['development'] != commitments['development'] or previous['final_test'] == commitments['final_test']:
            raise ValueError('New holdout needs unchanged development groups and a different frozen final set')
    write(output/'cohort-commitments.json', commitments)
    print(json.dumps({'stage': 'prepared', 'counts': cohort['provenance']['counts']}), flush=True)
    key = output/'store.key'; provision_key(key)
    store = LearningStore(output/'learning', key)
    store.add_grant(workspace_id='public-quality', rights=['private_training'],
        expires_at=time.time()+86400, evidence='Explicit public HelpSteer2 experiment under CC-BY-4.0; no customer data or shared-weight distribution')
    content = '\n'.join(json.dumps({'case_id': row['case_id'], 'group_id': row['group_id'],
        **row['input'], 'label': row['labels']['judgment']}) for row in cohort['development'])
    imported = import_dataset(store, workspace_id='public-quality', dataset_id='helpsteer-'+args.criterion,
        template=cohort['template'], content=content, format='jsonl', provenance='owner',
        evidence='NVIDIA HelpSteer2 external human reference labels; not authenticated RateLoop blind reviews')
    snapshot = store.create_snapshot('public-quality', cohort['template']['id'], cohort['template']['version'],
        train_fraction=.8, calibration_fraction=.1, purpose='private_training', dataset_version_ids=[imported['id']])
    base = measure(backend, cohort['final_test'], snapshot['calibration'])
    write(output/'base.json', base)
    unload(backend, args.device)
    print(json.dumps({'stage': 'baseline', 'balanced_agreement': base['balanced_agreement']}), flush=True)
    options = TrainOptions(method='lora', device=args.device, epochs=5, max_steps=200,
        batch_size=2, learning_rate=.0001, validation_fraction=.2, validation_interval=25,
        early_stopping_patience=3, min_validation_per_label=5)
    try:
        result = train_snapshot(store, snapshot['id'], 'public-quality', args.model_dir,
            output/'candidate', bundle_id='public-helpsteer-'+args.criterion, options=options)
    except ValidationQualityError:
        selection = json.loads((output/'candidate'/'selection-report.json').read_text())
        write(output/'report.json', {'schema_version': 'rateloop.public-quality-experiment.v1',
            'source': cohort['provenance'], 'base': {k:v for k,v in base.items() if k != 'model_manifest'},
            'selection': selection, 'candidate': None, 'quality_gate': False, 'activated': False,
            'reason': 'Validation rejected the candidate before artifact publication or candidate final-test scoring.'})
        print(json.dumps({'stage': 'validation_rejected', 'report': str(output/'report.json')}), flush=True)
        return
    print(json.dumps({'stage': 'trained', 'steps': result['training']['optimizerSteps'],
        'selected_step': result['training']['selection']['selectedStep']}), flush=True)
    candidate = GLiNERBackend(result['modelDir'], args.device)
    measured = measure(candidate, cohort['final_test'], snapshot['calibration'])
    write(output/'candidate.json', measured)
    unload(candidate, args.device)
    qid = 'judgment'
    report = {'schema_version': 'rateloop.public-quality-experiment.v1',
        'source': cohort['provenance'], 'training': result['training'],
        'base': {key: value for key, value in base.items() if key != 'model_manifest'},
        'candidate': {key: value for key, value in measured.items() if key != 'model_manifest'},
        'balanced_agreement_delta': measured['balanced_agreement']-base['balanced_agreement'],
        'false_approval_delta': measured['criteria'][qid]['false_approvals']-base['criteria'][qid]['false_approvals'],
        'base_manifest_sha256': hashlib.sha256(json.dumps(base['model_manifest'], sort_keys=True).encode()).hexdigest(),
        'candidate_manifest_sha256': hashlib.sha256(json.dumps(measured['model_manifest'], sort_keys=True).encode()).hexdigest(),
        'quality_gate': False, 'activated': False,
        'limits': 'One prespecified training recipe selected checkpoints on development validation only. Upstream validation was final test. Descriptive short-English public-data agreement, not independent deployment qualification. No retries tuned on final test.'}
    write(output/'report.json', report)
    print(json.dumps({'stage': 'complete', 'report': str(output/'report.json'),
        'balanced_agreement_delta': report['balanced_agreement_delta'],
        'false_approval_delta': report['false_approval_delta']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('train', 'test', 'model-dir', 'output'): parser.add_argument('--'+option, required=True)
    parser.add_argument('--criterion', choices=['helpfulness', 'correctness'], required=True)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    parser.add_argument('--development-groups', type=int, default=400)
    parser.add_argument('--test-groups', type=int, default=200)
    parser.add_argument('--previous-experiment', help='Freeze unused upstream-train tail as a new final holdout; prove previous development IDs unchanged')
    run(parser.parse_args())
