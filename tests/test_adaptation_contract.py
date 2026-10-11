"""Public exporter and consumer SDK share these synthetic commitment vectors."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rateloop_evaluator.backends import MANIFEST_NAME
from rateloop_evaluator.protocol import EvaluationRequest
from rateloop_evaluator.registrations import export_registration
from rateloop_evaluator.templates import custom_text_evaluation


@pytest.mark.parametrize('version,key', [(1, 'legacy'), (2, 'semantic')])
def test_registration_exports_shared_versioned_adaptation_commitments(tmp_path, version, key):
    fixture = json.loads((Path(__file__).parents[1]/'contracts/evaluator-adaptation.json').read_text())
    template = custom_text_evaluation('en', 'Does the answer meet the criterion?', 'Yes', 'No')
    request = EvaluationRequest(workspaceId='synthetic', caseId='fixture', idempotencyKey='fixture-key',
        modelBundleId='candidate', template=template, input={'text': 'Synthetic metadata only'})
    (tmp_path/MANIFEST_NAME).write_text(json.dumps({'files': fixture['files'],
        'source': {'baseWeightsSha256': '1'*64}, 'training': {'scoreAdapterVersion': version}}))
    manifest = {'languages': ['en'], 'template_commitments': [request.template_commitment()],
        'max_tokens': 512, 'calibrations': [], 'snapshot_id': 'frozen', 'model_id': 'fixture', 'model_revision': 'pinned'}
    registry = SimpleNamespace(get=lambda *args: {'manifest': manifest, 'artifact_root': str(tmp_path)},
        registration_policy=lambda *args: {'mode': 'shadow'})
    store = SimpleNamespace(load_snapshot=lambda *args: {'content_digest': 'a'*64})
    registration = export_registration(registry, store, 'synthetic', 'candidate', request)
    assert registration['adapterCommitment'] == fixture[key]
    assert registration['scoreCapability']['adapterVersion'] == version
    assert registration['trainingSnapshotCommitment'] == 'sha256:'+'a'*64
    assert all(row['calibrationId'] is None for row in registration['criteria'])
