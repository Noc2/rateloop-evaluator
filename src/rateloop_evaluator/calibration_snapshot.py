"""Calibration-only fitting with live rights, separate from certification evidence."""
from copy import deepcopy

from .backends import prepare_inference, render_input
from .calibration import apply_temperature, fit_binary_platt
from .learning import is_independent_reference
from .protocol import validate_reference_demonstration_isolation
from .quality import group_representatives


def fit_snapshot_calibrations(store, snapshot_id, workspace_id, backend, *, bundle_id, weights_sha256,
                              minimum_per_label=10):
    snapshot = store.load_snapshot(snapshot_id, workspace_id)
    rows = group_representatives(snapshot['calibration'])
    if not rows or any(not is_independent_reference(row) for row in snapshot['calibration']):
        raise PermissionError('Calibration requires independently collected blind human references')
    validate_reference_demonstration_isolation(snapshot['calibration'])
    other_groups = {row['group_id'] for part in ('train', 'test') for row in snapshot[part]}
    if any(row['group_id'] in other_groups for row in rows):
        raise ValueError('Calibration source groups overlap training or final evidence')
    predictions, eligible = [], []
    try:
        excluded = set()
        for row in snapshot['calibration']:
            store.load_snapshot(snapshot_id, workspace_id)
            prepared = prepare_inference(backend, render_input(row['input']), row['template']['questions'])
            if prepared.token_count > min(row['template']['maxTokens'], prepared.token_limit or row['template']['maxTokens']):
                excluded.add(row['group_id'])
        for row in rows:
            if row['group_id'] in excluded: continue
            store.load_snapshot(snapshot_id, workspace_id)
            prepared = prepare_inference(backend, render_input(row['input']), row['template']['questions'])
            if prepared.token_count > min(row['template']['maxTokens'], prepared.token_limit or row['template']['maxTokens']):
                continue
            eligible.append(row)
            predictions.append(prepared.predict())
        if not eligible:
            raise ValueError('Insufficient calibration data within the token limit')
        first = eligible[0]
        if any(row['template_commitment'] != first['template_commitment'] for row in rows):
            raise ValueError('Calibration requires one immutable task')
        for question in first['template']['questions']:
            if any(sum(row['labels'][question['id']] == label['id'] for row in eligible) < minimum_per_label
                   for label in question['labels']):
                raise ValueError('Insufficient calibration data: every label needs supported human source groups')
        artifacts = [fit_binary_platt([p[q['id']] for p in predictions], [r['labels'][q['id']] for r in eligible],
            model_bundle_id=bundle_id, template_commitment=first['template_commitment'], question_id=q['id'],
            language=first['template']['language'], example_ids=[r['group_id'] for r in eligible],
            model_weights_sha256=weights_sha256) for q in first['template']['questions']]
        store.load_snapshot(snapshot_id, workspace_id)
        return artifacts, {'calibration_group_count': len(rows), 'fitted_group_count': len(eligible),
            'overflow_group_count': len(rows)-len(eligible), 'quality_gate': False}
    finally:
        if hasattr(backend, 'unload'): backend.unload()


class CalibratedComparisonBackend:
    """Use exactly the registered mapping without reusing raw-score cache keys."""
    def __init__(self, backend, manifest):
        self.backend, self.bundle = backend, deepcopy(manifest)
        self.manifest = backend.manifest
    def prepare(self, text, questions):
        prepared = prepare_inference(self.backend, text, questions)
        self.manifest = self.backend.manifest
        original = prepared._predict
        def predict():
            scores = original()
            for artifact in self.bundle['calibrations']:
                qid = artifact['question_id']
                scores[qid] = apply_temperature(scores[qid], artifact, model_bundle_id=self.bundle['id'],
                    template_commitment=artifact['template_commitment'], question_id=qid, language=artifact['language'])
            return scores
        prepared._predict = predict
        return prepared
    def unload(self):
        if hasattr(self.backend, 'unload'): self.backend.unload()
