"""One content-free registration projection for ordinary and qualified exports."""
from datetime import datetime, timezone
import json
from pathlib import Path

from .backends import MANIFEST_NAME, GLINER_SCORE_CAPABILITY, tokenizer_commitment
from .protocol import commitment
from .templates import bundle_supports_template, is_custom_text_template


def export_registration(registry, store, workspace, bundle_id, request):
    record = registry.get(bundle_id, workspace); manifest = record['manifest']
    if request.workspaceId != workspace or request.modelBundleId != bundle_id or not bundle_supports_template(manifest, request.template):
        raise ValueError('Registration request does not match the signed bundle')
    model = json.loads((Path(record['artifact_root'])/MANIFEST_NAME).read_text())
    if model.get('training') and not model['source'].get('baseWeightsSha256'):
        raise ValueError('Trained model is missing its original weight digest')
    base_hash = model['source'].get('baseWeightsSha256', model['files'].get('model.safetensors'))
    if not base_hash:
        raise ValueError('Original model weight digest is unavailable')
    active = registry.registration_policy(bundle_id, workspace, request.template)
    calibrations = {c['question_id']: c for c in manifest['calibrations']
        if c['template_commitment'] == request.template_commitment() and c['language'] == request.template.language}
    expiry = (active.get('gate') or {}).get('valid_until')
    expiry = datetime.fromtimestamp(expiry, timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z') if expiry else None
    criteria = []
    for question in request.template.questions:
        calibration = calibrations.get(question.id) if expiry else None
        criteria.append({'questionId': question.id, 'labels': [label.id for label in question.labels],
            'calibrationId': calibration['id'] if calibration else None,
            'calibrationCommitment': commitment(calibration, 'rateloop.calibration.v1') if calibration else None,
            'calibrationExpiresAt': expiry if calibration else None})
    snapshot_digest = None
    if manifest.get('snapshot_id'):
        snapshot_digest = 'sha256:' + store.load_snapshot(manifest['snapshot_id'], workspace)['content_digest']
    registration = {'modelBundleId': bundle_id, 'templateCommitment': request.template_commitment(),
        'language': request.template.language, 'baseWeightsCommitment': 'sha256:' + base_hash,
        'adapterCommitment': commitment(model['files'], 'rateloop.adaptation.v1') if model.get('training') else None,
        'tokenizerCommitment': tokenizer_commitment(model), 'quantization': 'fp32',
        'trainingSnapshotCommitment': snapshot_digest,
        'evaluationReportCommitment': commitment(active, 'rateloop.deployment-evidence.v1'),
        'licenseManifestCommitment': commitment({'software': 'Apache-2.0', 'weights': model['source'].get('license', 'Apache-2.0'),
            'model': manifest['model_id'], 'revision': manifest['model_revision']}, 'rateloop.licenses.v1'),
        'maxTokens': manifest['max_tokens'], 'criteria': criteria, 'scoreCapability': dict(GLINER_SCORE_CAPABILITY)}
    if manifest.get('task_capability'):
        registration['taskCapability'] = manifest['task_capability']
    if is_custom_text_template(request.template):
        registration['template'] = request.template.model_dump()
    return registration
