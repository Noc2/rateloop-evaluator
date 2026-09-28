"""Endpoint auth failures suspend execution across inference and training consumers."""
from types import SimpleNamespace
import time

import httpx
import pytest

from rateloop_evaluator import cli
from rateloop_evaluator.connector import RateLoopConnector
from rateloop_evaluator.training_worker import TrainingWorker
from test_cli import initialized
from test_training_worker import runner, iso


@pytest.fixture
def coupled(runner):
    worker,behavior,authorization,job,template,store,registry,_,_=runner
    assert worker.run_once()["state"]=="training_completed"
    original_transport=worker.connector.client._transport
    remote={"workspaceId":"workspace-test","recipientApiKeyId":"api-1","revocationWatermark":1,
        "settings":{"mode":"shadow"},"grants":[],
        "consents":[{"consentId":"inference-consent","revision":1,"workspaceId":"workspace-test","apiKeyId":"api-1",
            "purpose":"ai_use","processingLocation":"rateloop_operated","modelFamilyId":"gliner25-multilingual",
            "modelBundleIds":["base"],"templateCommitments":[job["templateCommitment"]],"fields":["input"],
            "issuedAt":iso(time.time()-10),"expiresAt":None,"revokedAt":None}]}
    failure={"status":None,"endpoint":None}
    def transport(request):
        if failure["status"] and request.url.path.endswith(failure["endpoint"]):
            return httpx.Response(failure["status"],json={"error":"Private endpoint failure text"})
        if request.url.path.endswith("/grants"):
            remote["authorizationLease"]={"leaseId":"fresh-lease","issuedAt":iso(time.time()),"expiresAt":iso(time.time()+120),
                "revocationWatermark":remote["revocationWatermark"],"workspaceId":"workspace-test","recipientApiKeyId":"api-1"}
            return httpx.Response(200,json=remote)
        return original_transport.handle_request(request)
    worker.connector.client._transport=httpx.MockTransport(transport)
    worker.connector.sync_grants()
    scope={"workspace_id":"workspace-test","right":"ai_use","case_id":"scope-case","template_id":template.id,
        "fields":["input.text"],"model_bundle_id":"base","template_commitment":job["templateCommitment"]}
    with store.transaction() as database:
        inference_id=worker.connector._state(database)["consents"]["inference-consent"]["local_id"]
        training_id=worker._state(database)["permissions"][authorization["grantId"]]["localId"]
    foreign=[]
    for workspace,api_key in (("other-workspace","api-1"),("workspace-test","other-key")):
        grant=store.add_grant(workspace_id=workspace,rights=["ai_use"],expires_at=time.time()+3600,
            authorization_until=time.time()+120,case_ids=["foreign-case"],evidence="Independent connector permission")
        connector=RateLoopConnector(base_url="https://rateloop.example",api_key="unrelated-key",api_key_id=api_key,
            workspace_id=workspace,agent_id="agent-1",agent_version_id="version-1",learning=store,runtime=worker.connector.runtime)
        with store.transaction() as database:
            connector._state(database)["consents"]={"other-consent":{"local_id":grant["id"]}}
        foreign.append(grant)
        connector.close()
    return SimpleNamespace(worker=worker,behavior=behavior,remote=remote,failure=failure,transport=transport,
        scope=scope,inference_id=inference_id,training_id=training_id,foreign=foreign,job=job)


def restart(value):
    previous=value.worker
    root,local,store,registry=cli.state(SimpleNamespace(state_dir=previous.root))
    connector=RateLoopConnector(base_url="https://rateloop.example",api_key="scoped-operator-api-key",api_key_id="api-1",
        workspace_id="workspace-test",agent_id="agent-1",agent_version_id="version-1",learning=store,
        runtime=previous.connector.runtime,metadata_upload_enabled=True,transport=httpx.MockTransport(value.transport))
    value.worker=TrainingWorker(connector,registry,state_dir=root,worker_id="operator-1",model_dir=previous.model_dir,
        model_bundle_ids=["base"],heartbeat_seconds=1)
    previous.connector.close()


@pytest.mark.parametrize("status",[401,403])
@pytest.mark.parametrize("endpoint",["/grants","/training/workers/heartbeat","/training/jobs/claim"])
@pytest.mark.parametrize("first_renewal",["inference","training"])
def test_auth_failure_blocks_both_consumers_until_independent_fresh_proof(coupled,status,endpoint,first_renewal):
    value=coupled; worker=value.worker; store=worker.connector.learning
    value.failure.update(status=status,endpoint=endpoint)
    with pytest.raises(PermissionError,match="credential or scope") as rejected:
        if endpoint=="/grants": worker.connector.sync_grants()
        elif endpoint.endswith("workers/heartbeat"): worker.sync_permissions()
        else: worker.connector._request("POST",endpoint,json={"workerId":worker.worker_id})
    assert "Private endpoint" not in str(rejected.value)
    with pytest.raises(PermissionError):store.check_right(**value.scope)
    with pytest.raises(PermissionError,match="renewal"):worker.registry.get("candidate","workspace-test")
    with store.transaction() as database:
        state=worker.connector._state(database)
        assert state["mode"]=="paused" and state["authorization_lease"] is None
        for identity in (value.inference_id,value.training_id):
            assert database["grants"][identity]["authorization_until"]<=time.time()
            assert database["grants"][identity]["revoked_at"] is None
        assert database["lineage"]["candidate"]["retired_at"] is None
        for foreign in value.foreign:
            assert database["grants"][foreign["id"]]==foreign
    restart(value);worker=value.worker;store=worker.connector.learning
    with pytest.raises(PermissionError):store.check_right(**value.scope)
    with pytest.raises(PermissionError):worker.registry.get("candidate","workspace-test")
    value.failure["status"]=None
    if first_renewal=="inference":
        worker.connector.sync_grants()
        assert store.check_right(**value.scope)==[value.inference_id]
        with pytest.raises(PermissionError):worker.registry.get("candidate","workspace-test")
        worker.sync_permissions()
    else:
        worker.sync_permissions()
        assert worker.registry.get("candidate","workspace-test")
        with pytest.raises(PermissionError):store.check_right(**value.scope)
        with store.transaction() as database:assert worker.connector._state(database)["mode"]=="paused"
        worker.connector.sync_grants()
    assert store.check_right(**value.scope)==[value.inference_id]
    assert worker.registry.get("candidate","workspace-test")
    assert value.behavior["train_calls"]==1


@pytest.mark.parametrize("withdrawal",["inference_revoked","dataset_removed","already_poisoned"])
def test_auth_recovery_never_revives_owner_withdrawal_or_retired_weights(coupled,withdrawal):
    value=coupled;worker=value.worker;store=worker.connector.learning
    if withdrawal=="already_poisoned":
        worker.connector._revoke_mirrors("remote_credential_rejected")
        store.revoke_grant(value.training_id,"workspace-test")
    value.failure.update(status=403,endpoint="/training/workers/heartbeat")
    with pytest.raises(PermissionError):worker.sync_permissions()
    value.failure["status"]=None
    if withdrawal=="inference_revoked":
        value.remote["consents"][0]["revokedAt"]=iso(time.time())
        worker.connector.sync_grants()
        value.remote["consents"][0]["revokedAt"]=None
        with pytest.raises((PermissionError,ValueError)):worker.connector.sync_grants()
        with store.transaction() as database:assert database["grants"][value.inference_id]["revoked_at"] is not None
    elif withdrawal=="dataset_removed":
        value.behavior["permission"]=False;worker.sync_permissions()
        with pytest.raises(PermissionError,match="retired"):worker.registry.get("candidate","workspace-test")
        value.behavior["permission"]=True
        with pytest.raises((PermissionError,ValueError)):worker.sync_permissions()
        with store.transaction() as database:
            assert database["lineage"]["candidate"]["retired_at"] is not None
            assert not database["dataset_examples"]
    else:
        with pytest.raises(PermissionError):worker.connector.sync_grants()
        with pytest.raises(PermissionError):worker.sync_permissions()
        with store.transaction() as database:
            assert database["grants"][value.inference_id]["revoked_at"] is not None
            assert database["lineage"]["candidate"]["retired_at"] is not None
