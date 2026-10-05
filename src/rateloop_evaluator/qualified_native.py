"""Operator-signed, independently qualified, exact built-in native rubrics.

This capability is not arbitrary custom-text support. The exporter rechecks trusted
registry lineage and current held-out evidence; imported diagnostics cannot mint it.
"""
from __future__ import annotations

import base64
from collections import Counter
from copy import deepcopy
import math
import time

import rfc8785

from .calibration import apply_temperature
from .learning import is_independent_reference
from .protocol import commitment
from .quality import group_representatives, score_predictions
from .comparison import label_metrics
from .registrations import export_registration
from .templates import custom_text_evaluation

NATIVE_CAPABILITY = 'rateloop.evaluator.builtin-qualified-text.v1'
QUALIFICATION_SCHEMA = 'rateloop.native-qualification.v1'
SCOPE_FIELDS = {'rubricId', 'rubricVersion', 'language', 'templateCommitment', 'populationId',
                'routeId', 'routeVersion', 'evidencePolicyId', 'baselineBundleId', 'baselineModelCommitment', 'rolloutId'}


def builtin_template(rubric_id, language):
    questions = {
        'request_following': {'en': "Does this response address the user's request?",
                             'de': 'Geht diese Antwort auf die Anfrage ein?'},
        'source_faithfulness': {'en': 'Is this response supported by the supplied material?',
                              'de': 'Wird diese Antwort durch das bereitgestellte Material gestützt?'}}
    if rubric_id not in questions or language not in ('en', 'de'):
        raise ValueError('Unknown qualified built-in rubric or language')
    return custom_text_evaluation(language, questions[rubric_id][language],
                                  'Yes' if language == 'en' else 'Ja', 'No' if language == 'en' else 'Nein')


def validate_native_scope(scope, template):
    if not isinstance(scope, dict) or set(scope) != SCOPE_FIELDS or type(scope.get('rubricVersion')) is not int or scope.get('rubricVersion') != 1:
        raise ValueError('Native scope must pin the built-in rubric, population, route, baseline and rollout')
    if template != builtin_template(scope['rubricId'], scope['language']):
        raise ValueError('Qualified native scope requires the exact frozen built-in wording and labels')
    if scope['templateCommitment'] != commitment(template.model_dump(), 'rateloop.evaluator.template.v1'):
        raise ValueError('Qualified native template commitment differs')
    if type(scope['routeVersion']) is not int or scope['routeVersion'] < 1:
        raise ValueError('Native route must be versioned')
    for field in ('populationId', 'routeId', 'evidencePolicyId', 'baselineBundleId', 'rolloutId'):
        if not isinstance(scope[field], str) or not 1 <= len(scope[field]) <= 200:
            raise ValueError('Native qualification identities must be bounded strings')
    if (not isinstance(scope['baselineModelCommitment'], str) or not scope['baselineModelCommitment'].startswith('sha256:')
            or len(scope['baselineModelCommitment']) != 71 or any(c not in '0123456789abcdef' for c in scope['baselineModelCommitment'][7:])):
        raise ValueError('Pin the baseline bundle manifest commitment before final testing')
    if (scope['routeId'] != 'native-single-pass' or scope['routeVersion'] != 1
            or scope['evidencePolicyId'] != ('request-context-v1' if scope['rubricId'] == 'request_following' else 'supplied-material-v1')):
        raise ValueError('Unsupported native route or evidence policy')
    return deepcopy(scope)


def validate_native_registration_manifest(manifest):
    scope = manifest.get('native_scope')
    if scope is None:
        return
    from .protocol import Template
    template = Template.model_validate(manifest.get('template'))
    validate_native_scope(scope, template)
    if (manifest.get('task_capability') is not None or manifest.get('synthetic') is not False
            or not manifest.get('snapshot_id') or not manifest.get('calibrations')
            or manifest.get('template_commitments') != [scope['templateCommitment']]
            or manifest.get('languages') != [scope['language']]):
        raise ValueError('Qualified native candidates require independent lineage and one exact calibrated scope')
    policy = manifest.get('selective_policy', {})
    bounds = {'threshold': (.5, 1), 'minimum_coverage': (.5, 1), 'max_false_approval_rate': (0, .05), 'confidence': (.95, 1)}
    if (set(policy) != set(bounds) or any(type(policy[k]) not in (int, float) or not math.isfinite(policy[k])
            or not minimum <= policy[k] <= maximum for k, (minimum, maximum) in bounds.items())
            or any(policy[k] == 1 for k in ('threshold', 'minimum_coverage', 'confidence'))
            or policy['max_false_approval_rate'] == 0):
        raise ValueError('Native targets must be fixed before final testing')


def _independent_reference_diagnostics(snapshot, template):
    counts = {}
    for partition in ('calibration', 'test'):
        rows = group_representatives(snapshot[partition])
        for row in rows:
            humans = row.get('human_labels', [])
            if (not is_independent_reference(row) or len({h.get('annotator_id') for h in humans if h.get('annotator_id')}) < 2
                    or any(h.get('labels') != row['labels'] for h in humans)
                    or row.get('template_commitment') != commitment(template.model_dump(), 'rateloop.evaluator.template.v1')
                    or row.get('template') != template.model_dump()
                    or template == builtin_template('source_faithfulness', template.language) and not row.get('input', {}).get('evidence', '').strip()):
                raise PermissionError('Native qualification needs two agreeing authenticated blind reviewers on exact-scope references')
        references = Counter(r['labels']['judgment'] for r in rows)
        if len(rows) < 200 or any(references[label.id] < 20 for label in template.questions[0].labels):
            raise PermissionError('Native confidence requires at least 200 calibration/test groups and support for both classes')
        counts[partition] = {'sourceGroups': len(rows), 'referenceLabels': dict(references), 'minimumBlindReviewers': 2}
    return counts


def _baseline_diagnostics(registry, workspace, manifest, snapshot, evidence, baseline_evidence):
    scope = manifest['native_scope']
    baseline = registry.get(scope['baselineBundleId'], workspace)['manifest']
    if commitment(baseline, 'rateloop.bundle-manifest.v1') != scope['baselineModelCommitment']:
        raise PermissionError('Baseline model changed after candidate registration')
    if (not isinstance(baseline_evidence, dict) or baseline_evidence.get('bundle_id') != baseline['id']
            or baseline_evidence.get('template_commitment') != scope['templateCommitment']
            or baseline_evidence.get('language') != scope['language']
            or baseline_evidence.get('model_commitment') != scope['baselineModelCommitment']):
        raise ValueError('Baseline evidence does not bind the predeclared model and scope')
    observed = baseline_evidence.get('observed_at')
    if type(observed) not in (int, float) or not math.isfinite(observed) or observed < manifest['registered_at']:
        raise ValueError('Baseline final observations must follow operating-point registration')
    rows = group_representatives(snapshot['test'])
    expected = {r['evaluation_id'] for r in rows}
    predictions = baseline_evidence.get('rows', [])
    if len(predictions) != len(expected) or {r.get('evaluation_id') for r in predictions} != expected:
        raise ValueError('Baseline must cover the same complete final source groups')
    base_by_id = {r['evaluation_id']: r['raw_scores'] for r in predictions}
    candidate_by_id = {r['evaluation_id']: r['raw_scores'] for r in evidence['rows']}
    base = score_predictions(rows, [base_by_id[r['evaluation_id']] for r in rows])['criteria']['judgment']
    candidate = score_predictions(rows, [candidate_by_id[r['evaluation_id']] for r in rows])['criteria']['judgment']
    if (candidate['balanced_agreement'] < base['balanced_agreement']
            or candidate['false_approvals'] > base['false_approvals']
            or any(candidate['per_label'][label]['recall'] < base['per_label'][label]['recall'] for label in base['per_label'])):
        raise PermissionError('Native candidate regresses against the pinned baseline')
    return {'baseline': base, 'candidate': candidate,
            'baselineEvidenceCommitment': commitment(baseline_evidence, 'rateloop.native-baseline-evidence.v1')}


def export_qualified_native(registry, store, workspace, bundle_id, request, *, evidence,
                            baseline_evidence, rollback_bundle_id, now=None):
    current = time.time() if now is None else now
    record = registry.get(bundle_id, workspace); manifest = record['manifest']
    validate_native_registration_manifest(manifest)
    if not manifest.get('native_scope'):
        raise PermissionError('Native qualification scope must be signed before final testing')
    scope = validate_native_scope(manifest['native_scope'], request.template)
    active = registry.active(workspace, request.template_commitment(), request.template.language, now=current)
    if active['bundle_id'] != bundle_id or active['mode'] != 'selective':
        raise PermissionError('Native confidence requires this exact active qualified deployment')
    snapshot = store.load_snapshot(manifest['snapshot_id'], workspace, now=current)
    gate = registry._quality_gate(manifest, snapshot, evidence, now=current)
    if active.get('gate') != gate or gate['auto_approvals'] < 200 or gate['coverage'] < .5:
        raise PermissionError('Native evidence must match the active gate with at least 200 accepted cases and 50% coverage')
    references = _independent_reference_diagnostics(snapshot, request.template)
    comparison = _baseline_diagnostics(registry, workspace, manifest, snapshot, evidence, baseline_evidence)
    calibrations = [c for c in manifest['calibrations'] if c['template_commitment'] == scope['templateCommitment'] and c['language'] == scope['language']]
    test_rows = group_representatives(snapshot['test'])
    raw = {row['evaluation_id']: row['raw_scores'] for row in evidence['rows']}
    transformed = []
    for row in test_rows:
        transformed.append({cal['question_id']: apply_temperature(raw[row['evaluation_id']][cal['question_id']], cal,
            model_bundle_id=bundle_id, template_commitment=scope['templateCommitment'], question_id=cal['question_id'], language=scope['language'])
            for cal in calibrations})
    calibration_diagnostics = score_predictions(test_rows, transformed)['criteria']['judgment']
    candidate = comparison['candidate']
    if (calibration_diagnostics['expected_calibration_error'] >= .05
            or calibration_diagnostics['brier_score'] > candidate['brier_score'] + 1e-12
            or calibration_diagnostics['negative_log_likelihood'] > candidate['negative_log_likelihood'] + 1e-12):
        raise PermissionError('Held-out calibration diagnostics do not support native confidence display')
    if rollback_bundle_id == bundle_id:
        raise ValueError('Register a distinct rollback target')
    rollback = registry.get(rollback_bundle_id, workspace)
    if not rollback['manifest'].get('template_commitments'):
        raise ValueError('Rollback target has no registered scope')
    registration = export_registration(registry, store, workspace, bundle_id, request)
    registration['taskCapability'] = {'schemaVersion': NATIVE_CAPABILITY,
        'rubricId': scope['rubricId'], 'rubricVersion': 1}
    # Recheck lineage, activation and evidence validity immediately before signing.
    registry.get(bundle_id, workspace)
    if registry.active(workspace, request.template_commitment(), request.template.language, now=current) != active:
        raise PermissionError('Deployment changed while exporting native qualification')
    payload = {'schemaVersion': QUALIFICATION_SCHEMA,
        'registrationCommitment': commitment(registration, 'rateloop.evaluator-registration.v1'),
        'bundleEnvelope': record['envelope'], 'deploymentEvidence': active, 'calibrations': calibrations,
        'scope': scope, 'issuedAt': current, 'validUntil': gate['valid_until'], 'rollbackBundleId': rollback_bundle_id,
        'referenceDiagnostics': references, 'baselineDiagnostics': comparison,
        'calibrationDiagnostics': calibration_diagnostics,
        'limits': 'Exact low-consequence criterion/population only. AI never replaces required human review.'}
    envelope = {'manifest': payload, 'public_key': registry.public_key,
        'signature': base64.urlsafe_b64encode(registry._key.sign(rfc8785.dumps(payload))).decode()}
    exported = {'registration': registration, 'qualification': envelope}
    validate_qualified_native(exported, registry.public_key, now=current)
    return exported


def _validate_quality_diagnostics(metrics, counts):
    """Reject missing/non-numeric metrics and recompute class/confusion summaries."""
    if not isinstance(metrics, dict) or metrics.get('expected_label_counts') != counts:
        raise ValueError('Quality diagnostics must bind the exact reference population')
    total = sum(counts.values())
    if type(metrics.get('count')) is not int or metrics['count'] != total:
        raise ValueError('Quality sample count differs from its references')
    for name, maximum in (('brier_score', 2), ('negative_log_likelihood', None),
                           ('expected_calibration_error', 1), ('balanced_agreement', 1), ('agreement', 1)):
        value = metrics.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or maximum is not None and value > maximum:
            raise ValueError('Native quality diagnostics require finite numerical metrics')
    confusion = metrics.get('confusion', {})
    recomputed = label_metrics(confusion, counts)
    for label in counts:
        values = metrics.get('per_label', {}).get(label, {})
        for field in ('support', 'correct', 'predicted', 'abstentions'):
            if type(values.get(field)) is not int or values[field] < 0:
                raise ValueError('Class diagnostics require integer observation counts')
        for field in ('recall', 'precision'):
            value = values.get(field)
            if value is None and recomputed['per_label'][label][field] is None:
                continue
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Class diagnostics require finite numerical rates')
    if metrics.get('per_label') != recomputed['per_label'] or metrics['balanced_agreement'] != recomputed['balanced_agreement']:
        raise ValueError('Class diagnostics differ from the reference confusion matrix')
    correct = sum(confusion[label][label] for label in counts)
    if (type(metrics.get('false_approvals')) is not int or metrics['false_approvals'] != confusion['rejected']['approved']
            or type(metrics.get('false_rejections')) is not int or metrics['false_rejections'] != confusion['approved']['rejected']
            or type(metrics.get('correct')) is not int or metrics['correct'] != correct
            or metrics['agreement'] != correct/total):
        raise ValueError('Quality error counts differ from the confusion matrix')
    bins = metrics.get('calibration_bins')
    if not isinstance(bins, list) or len(bins) != 10:
        raise ValueError('Complete reliability bins are required')
    count = 0; expected_error = 0.
    for index, bucket in enumerate(bins):
        n = bucket.get('count')
        if (type(n) is not int or n < 0 or bucket.get('lower') != index/10 or bucket.get('upper') != (index+1)/10):
            raise ValueError('Invalid reliability-bin boundary or count')
        count += n
        for name in ('mean_raw_score', 'accuracy'):
            value = bucket.get(name)
            if (n == 0 and value is not None or n > 0 and
                    (type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1)):
                raise ValueError('Reliability bins require finite scores and observed rates')
        if n:
            if not math.isclose(bucket['accuracy'] * n, round(bucket['accuracy'] * n), abs_tol=1e-8):
                raise ValueError('Reliability-bin rate cannot represent fractional observations')
            expected_error += n * abs(bucket['accuracy'] - bucket['mean_raw_score'])
    if count != total or not math.isclose(metrics['expected_calibration_error'], expected_error/total, abs_tol=1e-9):
        raise ValueError('Reliability aggregate differs from the complete bins')


def validate_qualified_native(exported, trusted_public_key, *, now=None, revoked_registration_commitments=()):
    """Verify signed exact scope using an externally pinned operator key.

    A signature attests the operator's verified collection; it does not make
    external benchmark imports independent. Revocation is supplied by the host.
    """
    from datetime import datetime, timezone
    from .backends import GLINER_SCORE_CAPABILITY, MANIFEST_NAME, tokenizer_commitment
    from .calibration import false_approval_upper_bound, validate_calibration
    from .protocol import Template
    from .registry import verify_manifest
    current = time.time() if now is None else now
    if not isinstance(exported, dict) or set(exported) != {'registration', 'qualification'}:
        raise ValueError('Qualified native export requires registration and its signed qualification')
    registration = exported['registration']
    payload = verify_manifest(exported['qualification'], trusted_public_key)
    if payload.get('schemaVersion') != QUALIFICATION_SCHEMA:
        raise ValueError('Unknown native qualification schema')
    registration_hash = commitment(registration, 'rateloop.evaluator-registration.v1')
    if payload.get('registrationCommitment') != registration_hash or registration_hash in revoked_registration_commitments:
        raise PermissionError('Native registration changed or was revoked')
    scope = validate_native_scope(payload.get('scope'), Template.model_validate(registration.get('template')))
    if registration.get('taskCapability') != {'schemaVersion': NATIVE_CAPABILITY, 'rubricId': scope['rubricId'], 'rubricVersion': 1}:
        raise ValueError('Native capability is not this exact built-in rubric')
    bundle = verify_manifest(payload.get('bundleEnvelope', {}), trusted_public_key)
    validate_native_registration_manifest(bundle)
    if (bundle.get('native_scope') != scope or bundle.get('id') != registration.get('modelBundleId')
            or registration.get('templateCommitment') != scope['templateCommitment'] or registration.get('language') != scope['language']
            or registration.get('maxTokens') != 512 or registration.get('scoreCapability') != GLINER_SCORE_CAPABILITY
            or registration.get('quantization') != 'fp32' or not registration.get('trainingSnapshotCommitment')
            or registration.get('tokenizerCommitment') != tokenizer_commitment(bundle)):
        raise ValueError('Native registration, model and qualification identities differ')
    files = {key: value for key, value in bundle['files'].items() if key != MANIFEST_NAME}
    if registration.get('adapterCommitment') is not None and registration['adapterCommitment'] != commitment(files, 'rateloop.adaptation.v1'):
        raise ValueError('Native adaptation commitment differs from the signed model files')
    if registration.get('adapterCommitment') is None and registration.get('baseWeightsCommitment') != 'sha256:' + bundle['files'].get('model.safetensors', ''):
        raise ValueError('Native base weights differ from the signed model')
    deployment = payload.get('deploymentEvidence', {}); gate = deployment.get('gate') or {}
    if (deployment.get('mode') != 'selective' or deployment.get('bundle_id') != bundle['id']
            or deployment.get('workspace_id') != bundle.get('workspace_id')
            or deployment.get('template_commitment') != scope['templateCommitment'] or deployment.get('language') != scope['language']
            or registration.get('evaluationReportCommitment') != commitment(deployment, 'rateloop.deployment-evidence.v1')
            or gate.get('template_commitment') != scope['templateCommitment'] or gate.get('language') != scope['language']):
        raise ValueError('Native deployment evidence differs from the exact qualified scope')
    for value in (payload.get('issuedAt'), payload.get('validUntil'), gate.get('observed_at'), gate.get('valid_until')):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('Qualification validity timestamps must be finite')
    if (not 0 < gate['observed_at'] <= payload['issuedAt'] <= current < payload['validUntil'] == gate['valid_until']
            or gate['valid_until'] > gate['observed_at'] + 30*86400
            or gate['observed_at'] < bundle['registered_at']):
        raise PermissionError('Native qualification expired, future-dated or predates its registered operating point')
    policy = bundle['selective_policy']
    for key in ('threshold', 'minimum_coverage', 'confidence', 'max_false_approval_rate'):
        if gate.get(key) != policy.get(key) or type(gate.get(key)) not in (int, float) or not math.isfinite(gate[key]):
            raise ValueError('Native gate differs from its predeclared operating point')
    accepted, wrong, tested = (gate.get(key) for key in ('auto_approvals', 'wrong_approvals', 'test_groups'))
    if any(type(value) is not int for value in (accepted, wrong, tested)) or not 0 <= wrong <= accepted <= tested or accepted < 200:
        raise PermissionError('Native gate needs at least 200 independently audited acceptances')
    upper = false_approval_upper_bound(wrong, accepted, gate['confidence'])
    if (gate['confidence'] < .95 or gate['max_false_approval_rate'] > .05 or gate['minimum_coverage'] < .5
            or not .5 <= gate['threshold'] < 1 or not .95 <= gate['confidence'] < 1
            or not 0 < gate['max_false_approval_rate'] <= .05 or not .5 <= gate['minimum_coverage'] < 1
            or gate.get('coverage') != accepted/tested
            or gate['coverage'] < gate['minimum_coverage'] or upper > gate['max_false_approval_rate']
            or not math.isclose(gate.get('false_approval_upper_bound', -1), upper, rel_tol=1e-9, abs_tol=1e-12)):
        raise PermissionError('Native false-acceptance or coverage gate failed')
    if not isinstance(payload.get('rollbackBundleId'), str) or payload['rollbackBundleId'] == bundle['id']:
        raise ValueError('A distinct rollback target is required')
    references = payload.get('referenceDiagnostics', {})
    for part in ('calibration', 'test'):
        reference = references.get(part, {})
        counts = reference.get('referenceLabels', {})
        if (type(reference.get('sourceGroups')) is not int or reference['sourceGroups'] < 200
                or reference.get('minimumBlindReviewers') != 2 or set(counts) != {'approved', 'rejected'}
                or any(type(value) is not int or value < 20 for value in counts.values())
                or sum(counts.values()) != reference['sourceGroups']):
            raise PermissionError('Native references lack independent class/group support')
    if references['test']['sourceGroups'] != tested:
        raise ValueError('Native reference support differs from the gate population')
    calibrations = payload.get('calibrations', [])
    if calibrations != bundle['calibrations'] or len(calibrations) != 1:
        raise ValueError('Native calibration artifacts differ from the signed bundle')
    calibration = calibrations[0]; validate_calibration(calibration)
    expiry = datetime.fromtimestamp(payload['validUntil'], timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')
    if (calibration['model_bundle_id'] != bundle['id'] or calibration['template_commitment'] != scope['templateCommitment']
            or calibration['language'] != scope['language'] or calibration['question_id'] != 'judgment'
            or calibration['model_weights_sha256'] != bundle['files']['model.safetensors']
            or set(calibration['label_ids']) != {'approved', 'rejected'}
            or calibration['sample_count'] != references['calibration']['sourceGroups']
            or gate.get('calibration_ids') != [calibration['id']]
            or deployment.get('calibration_ids') != [calibration['id']]
            or registration.get('criteria') != [{'questionId': 'judgment', 'labels': ['approved', 'rejected'],
                'calibrationId': calibration['id'], 'calibrationCommitment': commitment(calibration, 'rateloop.calibration.v1'),
                'calibrationExpiresAt': expiry}]):
        raise ValueError('Native calibration is not bound to the exact model/rubric/reference population')
    comparison = payload.get('baselineDiagnostics', {})
    baseline = comparison.get('baseline', {}); candidate = comparison.get('candidate', {})
    diagnostics = payload.get('calibrationDiagnostics', {})
    for metrics in (baseline, candidate, diagnostics):
        _validate_quality_diagnostics(metrics, references['test']['referenceLabels'])
    try:
        if (candidate['count'] != tested or baseline['count'] != tested or diagnostics['count'] != tested
                or candidate['balanced_agreement'] < baseline['balanced_agreement']
                or candidate['false_approvals'] > baseline['false_approvals']
                or any(candidate['per_label'][label]['recall'] < baseline['per_label'][label]['recall'] for label in ('approved', 'rejected'))
                or diagnostics['expected_calibration_error'] >= .05
                or diagnostics['brier_score'] > candidate['brier_score'] + 1e-12
                or diagnostics['negative_log_likelihood'] > candidate['negative_log_likelihood'] + 1e-12):
            raise PermissionError('Native baseline or held-out calibration diagnostics failed')
    except (KeyError, TypeError) as exc:
        raise ValueError('Native quality diagnostics are incomplete') from exc
    return deepcopy(payload)
