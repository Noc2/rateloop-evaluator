"""Synthetic mechanics fixtures: these are not independent human qualification."""
import base64
from copy import deepcopy
import json
import time

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest
import rfc8785

from rateloop_evaluator.calibration import fit_temperature
from rateloop_evaluator.protocol import EvaluationRequest, commitment
from rateloop_evaluator.qualified_native import (
    builtin_template, export_qualified_native, validate_native_scope, validate_qualified_native,
    validate_native_registration_manifest,
)
from rateloop_evaluator.registry import BundleRegistry


@pytest.fixture
def qualified_fixture(tmp_path):
    now = time.time(); template = builtin_template('request_following', 'en')
    template_hash = commitment(template.model_dump(), 'rateloop.evaluator.template.v1')
    files = {'model.safetensors': 'c'*64, 'tokenizer.json': 'd'*64}
    model = {'files': files, 'source': {'license': 'Apache-2.0'}}
    (tmp_path/'rateloop-model.json').write_text(json.dumps(model))
    def source_row(prefix, i):
        label = 'approved' if i % 2 else 'rejected'
        return {'evaluation_id': f'{prefix}-{i:04}', 'group_id': f'{prefix}-group-{i}',
            'case_id': f'{prefix}-case-{i}', 'input': {'text': f'{prefix} authored synthetic case {i}'},
            'template': template.model_dump(), 'template_commitment': template_hash, 'labels': {'judgment': label},
            'human_labels': [{'annotator_id': reviewer, 'independent_human': True, 'exposed_to_ai': False,
                             'labels': {'judgment': label}, 'quarantine_reasons': []} for reviewer in ('fixture-a', 'fixture-b')]}
    snapshot = {'purpose': 'private_training', 'content_digest': 'e'*64,
        'train': [source_row('train', i) for i in range(2)],
        'calibration': [source_row('cal', i) for i in range(200)],
        'test': [source_row('test', i) for i in range(400)]}
    def scores(rows):
        return [{'approved': .99 if r['labels']['judgment'] == 'approved' else .01,
                 'rejected': .01 if r['labels']['judgment'] == 'approved' else .99} for r in rows]
    calibration = fit_temperature(scores(snapshot['calibration']), [r['labels']['judgment'] for r in snapshot['calibration']],
        model_bundle_id='qualified', template_commitment=template_hash, question_id='judgment', language='en',
        example_ids=[r['group_id'] for r in snapshot['calibration']], model_weights_sha256='c'*64)
    baseline = {'id': 'base', 'template_commitments': [template_hash], 'files': files, 'languages': ['en']}
    scope = {'rubricId': 'request_following', 'rubricVersion': 1, 'language': 'en', 'templateCommitment': template_hash,
        'populationId': 'synthetic-fixture-never-publish', 'routeId': 'native-single-pass', 'routeVersion': 1,
        'evidencePolicyId': 'request-context-v1', 'baselineBundleId': 'base',
        'baselineModelCommitment': commitment(baseline, 'rateloop.bundle-manifest.v1'), 'rolloutId': 'fixture'}
    policy = {'threshold': .95, 'max_false_approval_rate': .05, 'minimum_coverage': .5, 'confidence': .95}
    manifest = {'id': 'qualified', 'workspace_id': 'workspace', 'model_id': 'synthetic-test-only', 'model_revision': 'a'*40,
        'registered_at': now-120, 'synthetic': False, 'snapshot_id': 'snapshot',
        'template': template.model_dump(), 'template_commitments': [template_hash], 'languages': ['en'],
        'calibrations': [calibration], 'files': files, 'max_tokens': 512, 'selective_policy': policy, 'native_scope': scope}
    evidence = {'observed_at': now-60, 'valid_until': now+3600, 'synthetic': False,
        'template_commitment': template_hash, 'language': 'en', **policy,
        'rows': [{'evaluation_id': r['evaluation_id'], 'raw_scores': {'judgment': score}}
                 for r, score in zip(snapshot['test'], scores(snapshot['test']))]}
    gate = BundleRegistry._quality_gate(manifest, snapshot, evidence, now=now)
    deployment = {'bundle_id': 'qualified', 'workspace_id': 'workspace', 'template_commitment': template_hash,
        'language': 'en', 'mode': 'selective', 'promoted_at': now-30, 'gate': gate, 'calibration_ids': [calibration['id']]}
    class Store:
        def load_snapshot(self, *_args, **_kwargs): return deepcopy(snapshot)
    class Registry:
        _quality_gate = staticmethod(BundleRegistry._quality_gate)
        def __init__(self):
            self._key = Ed25519PrivateKey.generate()
            self.public_key = base64.urlsafe_b64encode(self._key.public_key().public_bytes(
                serialization.Encoding.Raw, serialization.PublicFormat.Raw)).decode()
        def sign(self, value):
            return {'manifest': deepcopy(value), 'public_key': self.public_key,
                'signature': base64.urlsafe_b64encode(self._key.sign(rfc8785.dumps(value))).decode()}
        def get(self, bundle, _workspace, **_kwargs):
            value = manifest if bundle == 'qualified' else baseline
            return {'manifest': deepcopy(value), 'envelope': self.sign(value), 'artifact_root': str(tmp_path)}
        def active(self, *_args, **_kwargs): return deepcopy(deployment)
        def registration_policy(self, *_args, **_kwargs): return deepcopy(deployment)
    registry = Registry()
    request = EvaluationRequest.model_validate({'idempotencyKey': 'fixture-request', 'workspaceId': 'workspace', 'caseId': 'case',
        'modelBundleId': 'qualified', 'template': template.model_dump(), 'input': {'text': 'synthetic'}})
    baseline_evidence = {'bundle_id': 'base', 'template_commitment': template_hash, 'language': 'en',
        'model_commitment': scope['baselineModelCommitment'], 'observed_at': now-60, 'rows': deepcopy(evidence['rows'])}
    return registry, Store(), request, manifest, snapshot, evidence, baseline_evidence, now


def export(fixture):
    registry, store, request, _, _, evidence, baseline, now = fixture
    return export_qualified_native(registry, store, 'workspace', 'qualified', request,
        evidence=evidence, baseline_evidence=baseline, rollback_bundle_id='base', now=now)


def test_export_recomputes_trusted_gate_and_binds_registration_model_scope(qualified_fixture):
    exported = export(qualified_fixture)
    registry, _, request, _, _, _, _, now = qualified_fixture
    verified = validate_qualified_native(exported, registry.public_key, now=now)
    assert verified['deploymentEvidence']['gate']['auto_approvals'] == 200
    assert verified['referenceDiagnostics']['calibration']['minimumBlindReviewers'] == 2
    assert exported['registration']['taskCapability']['rubricId'] == 'request_following'
    assert exported['registration']['templateCommitment'] == request.template_commitment()
    assert 'authored synthetic case' not in json.dumps(exported)


def test_signature_registration_tampering_expiry_and_revocation_fail(qualified_fixture):
    exported = export(qualified_fixture); registry = qualified_fixture[0]; now = qualified_fixture[-1]
    with pytest.raises(ValueError, match='trusted key'):
        validate_qualified_native(exported, 'attacker', now=now)
    changed = deepcopy(exported); changed['registration']['quantization'] = 'int4'
    with pytest.raises(PermissionError, match='changed'):
        validate_qualified_native(changed, registry.public_key, now=now)
    with pytest.raises(PermissionError, match='expired'):
        validate_qualified_native(exported, registry.public_key, now=now+4000)
    with pytest.raises(PermissionError, match='revoked'):
        validate_qualified_native(exported, registry.public_key, now=now,
            revoked_registration_commitments=[exported['qualification']['manifest']['registrationCommitment']])


def test_public_imports_and_single_human_references_cannot_export(qualified_fixture):
    snapshot = qualified_fixture[4]
    snapshot['calibration'][0]['source_kind'] = 'dataset'
    with pytest.raises(PermissionError, match='independently collected'):
        export(qualified_fixture)
    snapshot['calibration'][0].pop('source_kind')
    snapshot['calibration'][0]['human_labels'].pop()
    with pytest.raises(PermissionError, match='two agreeing'):
        export(qualified_fixture)


def test_a_signed_claim_cannot_override_recomputed_bounds_or_exact_calibration(qualified_fixture):
    exported = export(qualified_fixture); registry = qualified_fixture[0]; now = qualified_fixture[-1]
    changed = deepcopy(exported); payload = changed['qualification']['manifest']
    payload['deploymentEvidence']['gate']['wrong_approvals'] = 20
    changed['registration']['evaluationReportCommitment'] = commitment(payload['deploymentEvidence'], 'rateloop.deployment-evidence.v1')
    payload['registrationCommitment'] = commitment(changed['registration'], 'rateloop.evaluator-registration.v1')
    changed['qualification'] = registry.sign(payload)
    with pytest.raises(PermissionError, match='false-acceptance'):
        validate_qualified_native(changed, registry.public_key, now=now)
    changed = deepcopy(exported); payload = changed['qualification']['manifest']
    payload['calibrations'][0]['temperature'] = 1
    changed['qualification'] = registry.sign(payload)
    with pytest.raises(ValueError, match='artifacts differ'):
        validate_qualified_native(changed, registry.public_key, now=now)


def test_native_scope_is_fixed_before_observation_and_never_generic_custom(qualified_fixture):
    _, _, request, manifest, *_ = qualified_fixture
    scope = deepcopy(manifest['native_scope'])
    scope['evidencePolicyId'] = 'fetch-whatever'
    with pytest.raises(ValueError, match='Unsupported native route'):
        validate_native_scope(scope, request.template)
    custom = deepcopy(manifest); custom['task_capability'] = {'schemaVersion': 'rateloop.evaluator.custom-binary-text.v1'}
    with pytest.raises(ValueError, match='independent lineage'):
        validate_native_registration_manifest(custom)
    manifest['native_scope']['templateCommitment'] = 'sha256:'+'b'*64
    with pytest.raises(ValueError, match='commitment differs'):
        export(qualified_fixture)


def test_baseline_regression_blocks_export(qualified_fixture):
    candidate = qualified_fixture[5]['rows']
    # Still meets acceptance risk/coverage? Changing one negative to positive has
    # enough sample support, but degrades negative recall and fails the baseline.
    candidate[0]['raw_scores']['judgment'] = {'approved': .99, 'rejected': .01}
    registry = qualified_fixture[0]
    gate = BundleRegistry._quality_gate(qualified_fixture[3], qualified_fixture[4], qualified_fixture[5], now=qualified_fixture[-1])
    old = registry.active()
    old['gate'] = gate
    registry.active = lambda *_args, **_kwargs: deepcopy(old)
    registry.registration_policy = registry.active
    with pytest.raises(PermissionError, match='regresses'):
        export(qualified_fixture)
