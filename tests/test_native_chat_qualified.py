"""Synthetic signed-envelope mechanics; no real qualification is asserted."""
from copy import deepcopy
from datetime import datetime, timezone
import time

from fastapi import HTTPException
import httpx
import pytest

from rateloop_evaluator.native_chat_pool import NativeChatPool, evaluate_native_job
from rateloop_evaluator.protocol import EvaluationRequest
from test_qualified_native import qualified_fixture, export


class Backend:
    manifest = {'files': {'model.safetensors': 'c'*64, 'tokenizer.json': 'd'*64}}
    def __init__(self): self.calls = 0
    def count_tokens(self, *_): return 20
    def predict(self, *_):
        self.calls += 1
        return {'judgment': {'approved': .99, 'rejected': .01}}


def job_and_config(fixture):
    exported = export(fixture)
    request = fixture[2].model_copy(update={'workspaceId': 'native-user-workspace'})
    expires = datetime.fromtimestamp(time.time()+90, timezone.utc).isoformat().replace('+00:00', 'Z')
    job = {'workerId': 'rateloop-native-chat-v1', 'workspaceId': request.workspaceId,
        'jobId': 'native-job', 'leaseToken': 'lease', 'leaseExpiresAt': expires,
        'modelBundleId': request.modelBundleId, 'inputCommitment': request.input_commitment(),
        'templateCommitment': request.template_commitment(), 'agentId': 'agent', 'agentVersionId': 'version',
        'content': {'request': request.model_dump(), 'agentId': 'agent', 'agentVersionId': 'version', 'retainForTraining': False},
        'baseRegistration': exported['registration'], 'qualification': exported['qualification']}
    config = {'trustedPublicKey': fixture[0].public_key, 'registrations': [exported]}
    return job, config


def test_qualified_native_returns_exact_probabilities_but_stays_advisory(qualified_fixture):
    job, config = job_and_config(qualified_fixture); backend = Backend()
    result = evaluate_native_job(job, backend=backend, qualified_native=config)
    assert backend.calls == 1
    assert result['criteria'][0]['probabilities']['approved'] > .99
    assert result['criteria'][0]['calibrationId'] == job['baseRegistration']['criteria'][0]['calibrationId']
    assert result['outcome'] == 'uncertain' and result['abstainReason'] == 'shadow_only'


def test_trust_pin_revocation_and_custom_wording_block_before_inference(qualified_fixture):
    job, config = job_and_config(qualified_fixture); backend = Backend()
    with pytest.raises(PermissionError, match='trust pin'):
        evaluate_native_job(job, backend=backend)
    changed = deepcopy(job); req = changed['content']['request']
    req['template']['questions'][0]['text'] = 'Different custom question?'
    changed['templateCommitment'] = EvaluationRequest.model_validate(req).template_commitment()
    changed['inputCommitment'] = EvaluationRequest.model_validate(req).input_commitment()
    with pytest.raises(PermissionError, match='Custom wording'):
        evaluate_native_job(changed, backend=backend, qualified_native=config)
    config['revokedRegistrationCommitments'] = [job['qualification']['manifest']['registrationCommitment']]
    with pytest.raises(PermissionError, match='revoked'):
        evaluate_native_job(job, backend=backend, qualified_native=config)
    assert backend.calls == 0


def test_revocation_during_inference_prevents_receipt_release(qualified_fixture):
    job, config = job_and_config(qualified_fixture)
    class Revoking(Backend):
        def predict(self, *_):
            config['revokedRegistrationCommitments'] = [job['qualification']['manifest']['registrationCommitment']]
            return super().predict()
    with pytest.raises((HTTPException, PermissionError)):
        evaluate_native_job(job, backend=Revoking(), qualified_native=config)


@pytest.mark.parametrize('qualified_fixture', ['source_faithfulness'], indirect=True)
def test_qualified_source_rubric_never_guesses_without_supplied_material(qualified_fixture):
    job, config = job_and_config(qualified_fixture); backend = Backend()
    with pytest.raises(ValueError, match='requires supplied material'):
        evaluate_native_job(job, backend=backend, qualified_native=config)
    assert backend.calls == 0
    job['content']['request']['input']['evidence'] = 'An explicit synthetic source.'
    job['inputCommitment'] = EvaluationRequest.model_validate(job['content']['request']).input_commitment()
    result = evaluate_native_job(job, backend=backend, qualified_native=config)
    assert backend.calls == 1 and result['criteria'][0]['probabilities']


def test_expired_or_revoked_qualified_registration_withdraws_while_base_stays(qualified_fixture):
    import json
    job, config = job_and_config(qualified_fixture)
    config['revokedRegistrationCommitments'] = [job['qualification']['manifest']['registrationCommitment']]
    base = {'modelBundleId': 'base', 'taskCapability': {'schemaVersion': 'rateloop.evaluator.custom-binary-text.v1'}}
    seen = []
    def transport(request):
        seen.append(json.loads(request.content)); return httpx.Response(200, json={'job': None})
    pool = NativeChatPool(secret='s'*32, base_url='https://www.rateloop.ai',
        bundles=[base, job['baseRegistration']], backend=Backend(), qualified_native=config,
        transport=httpx.MockTransport(transport))
    assert pool.run_once() == {'state': 'idle'}
    assert seen[0]['bundles'] == [base]
    pool.close()


def test_qualified_runtime_rechecks_actual_loaded_checkpoint(qualified_fixture):
    job, config = job_and_config(qualified_fixture)
    backend = Backend(); backend.manifest = {'files': {'model.safetensors': 'f'*64, 'tokenizer.json': 'd'*64}}
    with pytest.raises(PermissionError, match='loaded public model'):
        evaluate_native_job(job, backend=backend, qualified_native=config)
    assert backend.calls == 0
