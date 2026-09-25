from copy import deepcopy
import time

import pytest

from test_connector import setup, iso, evaluate, export_labels
from rateloop_evaluator.protocol import commitment, utc_now
from rateloop_evaluator.templates import overall_approval
from rateloop_evaluator.connector import AuthorizationLeaseRejected


def durable(remote, req, now):
    remote["grants"]=[]
    remote["consents"]=[{"consentId":"permission-1","revision":1,"workspaceId":req.workspaceId,"apiKeyId":"api-key-1",
        "purpose":"private_learning","processingLocation":"rateloop_operated","modelFamilyId":"gliner25-multilingual",
        "modelBundleIds":[req.modelBundleId],"templateCommitments":[req.template_commitment()],
        "fields":["input","context","evidence","human_labels"],"issuedAt":iso(now-10),"expiresAt":None,"revokedAt":None}]
    remote["authorizationLease"]={"leaseId":"lease-1","issuedAt":iso(now-1),"expiresAt":iso(now+800),
        "revocationWatermark":1,"workspaceId":req.workspaceId,"recipientApiKeyId":"api-key-1"}


def test_renewable_authorization_preserves_lineage_but_never_revives_withdrawal(setup):
    connector,req,learning,_,remote,_,_,_=setup
    now=time.time(); durable(remote,req,now)
    connector.sync_grants(now=now)
    scope=dict(workspace_id=req.workspaceId,right="private_training",case_id=req.caseId,template_id=req.template.id,
        fields=["input.text"],model_bundle_id=req.modelBundleId,template_commitment=req.template_commitment())
    original=learning.check_right(**scope,now=now)
    with pytest.raises(PermissionError): learning.check_right(**scope,now=now+801)
    later=now+86400
    remote["authorizationLease"].update(leaseId="lease-next-day",issuedAt=iso(later-1),expiresAt=iso(later+800))
    connector.sync_grants(now=later)
    assert learning.check_right(**scope,now=later)==original
    remote["consents"][0]["revokedAt"]=iso(later)
    connector.sync_grants(now=later)
    remote["consents"][0]["revokedAt"]=None
    with pytest.raises((PermissionError,ValueError)):
        connector.sync_grants(now=later)


def test_changed_scope_and_excessive_leases_fail_closed(setup):
    connector,req,learning,_,remote,_,_,_=setup
    now=time.time(); durable(remote,req,now); connector.sync_grants(now=now)
    remote["consents"][0]["modelBundleIds"].append("another-model")
    with pytest.raises(PermissionError,match="changed"): connector.sync_grants(now=now)
    remote["consents"][0]["modelBundleIds"].pop()
    remote["authorizationLease"]["expiresAt"]=iso(now+901)
    with pytest.raises(PermissionError,match="lease"): connector.sync_grants(now=now)


@pytest.mark.parametrize("reason,changes",[
    ("workspace_mismatch",lambda now:{"workspaceId":"private-wrong-workspace"}),
    ("recipient_mismatch",lambda now:{"recipientApiKeyId":"private-wrong-recipient"}),
    ("watermark_mismatch",lambda now:{"revocationWatermark":987654321}),
    ("invalid_issue_time",lambda now:{"issuedAt":iso(0)}),
    ("future_issue",lambda now:{"issuedAt":iso(now+5.001)}),
    ("invalid_duration",lambda now:{"expiresAt":iso(now-1)}),
    ("expired",lambda now:{"expiresAt":iso(now)}),
    ("excessive_duration",lambda now:{"expiresAt":iso(now+900)}),
])
def test_authorization_lease_rejections_have_fixed_private_safe_reasons(setup,reason,changes):
    connector,req,learning,_,remote,_,_,_=setup
    now=float(int(time.time()));durable(remote,req,now)
    with learning.transaction() as state:
        original_grants=deepcopy(state["grants"])
    remote["authorizationLease"].update(changes(now))
    with pytest.raises(AuthorizationLeaseRejected) as rejected:
        connector.sync_grants(now=now)
    assert rejected.value.reason==reason
    assert str(rejected.value)=="Worker authorization lease rejected: "+reason
    with learning.transaction() as state:
        assert state["grants"]==original_grants
        assert connector._state(state)["last_failure"]=="invalid_remote_grant_state"


def test_authorization_lease_exact_issue_and_duration_use_conservative_local_expiry(setup):
    connector,req,learning,_,remote,_,_,_=setup
    now=float(int(time.time()));durable(remote,req,now)
    remote["authorizationLease"].update(issuedAt=iso(now),expiresAt=iso(now+900))
    connector.sync_grants(now=now)
    with learning.transaction() as state:
        assert [g["authorization_until"] for g in state["grants"].values() if g["authorization_until"] is not None]==[now+895]


def test_authorization_lease_diagnostics_reject_unrecognized_free_text():
    with pytest.raises(ValueError,match="Unknown authorization lease rejection reason"):
        AuthorizationLeaseRejected("untrusted server content")


def test_source_snapshot_requires_current_lease_even_with_another_active_grant(setup):
    connector,req,learning,_,remote,_,_,_=setup
    now=time.time(); durable(remote,req,now); connector.sync_grants(now=now)
    for index in range(3):
        item=req.model_copy(deep=True)
        item.caseId=f"case-{index}"; item.sourceGroupId=f"group-{index}"; item.input.text+=str(index)
        evaluate(learning,item)
        learning.add_feedback(workspace_id=item.workspaceId,evaluation_id=item.input_commitment(),input_commitment=item.input_commitment(),
            template_commitment=item.template_commitment(),annotator_id="human-1",labels={"tone":"suitable"},exposed_to_ai=False,independent_human=True)
    snapshot=learning.create_snapshot(req.workspaceId,req.template.id,req.template.version)
    learning.add_grant(workspace_id=req.workspaceId,rights=["private_training"],expires_at=now+100000,evidence="Other unrelated grant")
    with pytest.raises(PermissionError,match="renewal"):
        learning.load_snapshot(snapshot["id"],req.workspaceId,now=now+801)


def test_frozen_blind_human_answer_remains_independent_after_reveal(setup):
    connector,req,learning,_,remote,behavior,_,_=setup
    connector.sync_grants()
    connector.run_with_audit(req,lambda r:evaluate(learning,r),__import__("test_connector").review_context(),frozen_question_hash="sha256:"+"e"*64)
    behavior["labels"]=export_labels(connector,req,remote)
    item=behavior["labels"]["items"][0]
    bindings={k:"sha256:"+"c"*64 for k in ("sourceContentHash","suggestedContentHash","frozenQuestionHash")}
    with learning.transaction() as db:
        connector._state(db)["audits"][req.input_commitment()].update(bindings)
    item.update(bindings,questionId="tone")
    item["audit"].update(blindingAssurance="server_enforced",independent=True,reviewFrozenAt=utc_now(),resultsReleasedAt=utc_now(),aiExposed=True)
    connector.release_result(req.input_commitment())
    body={k:v for k,v in behavior["labels"].items() if k!="exportDigest"}
    behavior["labels"]["exportDigest"]=commitment(body,"rateloop.product-evaluator.v2")
    result=connector.fetch_and_import_labels("server-grant-1",question_id="tone",template_commitment=req.template_commitment(),
        outcome_labels={"positive":"suitable","negative":"unsuitable"})
    assert result["imported"]==1 and not result["rejected"]


def test_exposure_before_frozen_answer_is_rejected(setup):
    connector,req,learning,_,remote,behavior,_,_=setup
    connector.sync_grants()
    connector.run_with_audit(req,lambda r:evaluate(learning,r),__import__("test_connector").review_context(),frozen_question_hash="sha256:"+"e"*64)
    behavior["labels"]=export_labels(connector,req,remote)
    item=behavior["labels"]["items"][0]
    item["audit"].update(blindingAssurance="server_enforced",independent=True,reviewFrozenAt=utc_now(),resultsReleasedAt=iso(time.time()-60))
    body={k:v for k,v in behavior["labels"].items() if k!="exportDigest"}
    behavior["labels"]["exportDigest"]=commitment(body,"rateloop.product-evaluator.v2")
    result=connector.fetch_and_import_labels("server-grant-1",question_id="tone",template_commitment=req.template_commitment(),
        outcome_labels={"positive":"suitable","negative":"unsuitable"})
    assert result["imported"]==0 and len(result["rejected"])==1


def test_snapshot_cli_and_store_share_exact_multilingual_scope(setup,monkeypatch):
    from argparse import Namespace
    from rateloop_evaluator import cli
    connector,req,learning,_,_,_,_,_=setup
    learning.add_grant(workspace_id=req.workspaceId,rights=["private_training"],expires_at=time.time()+600,evidence="Synthetic translated templates")
    for language in ("en","de"):
        for index in range(3):
            item=req.model_copy(deep=True);item.template=overall_approval(language)
            item.caseId=f"case-{language}-{index}";item.sourceGroupId=item.caseId;item.input.text+=item.caseId
            evaluate(learning,item)
            learning.add_feedback(workspace_id=req.workspaceId,evaluation_id=item.input_commitment(),input_commitment=item.input_commitment(),
                template_commitment=item.template_commitment(),annotator_id="independent-human",labels={"overall_approval":"approved"},
                exposed_to_ai=False,independent_human=True)
    with pytest.raises(ValueError,match="conflicting"):
        learning.create_snapshot(req.workspaceId,"customer-reply-approval",1)
    monkeypatch.setattr(cli,"state",lambda _: (learning.root,{"workspaceId":req.workspaceId},learning,None))
    for language in ("en","de"):
        digest=commitment(overall_approval(language).model_dump(),"rateloop.evaluator.template.v1")
        result=cli.run(Namespace(command="snapshot",template="customer-reply-approval",version=1,purpose="private_training",template_commitment=digest))
        snapshot=learning.load_snapshot(result["snapshotId"],req.workspaceId)
        assert result["groups"]==3
        assert {row["template"]["language"] for part in ("train","calibration","test") for row in snapshot[part]}=={language}


@pytest.mark.parametrize("mode",["off","paused"])
def test_pause_restart_resume_preserves_durable_grant_and_snapshot_lineage(setup,mode):
    from rateloop_evaluator.connector import RateLoopConnector
    from rateloop_evaluator.learning import LearningStore
    from rateloop_evaluator.storage import RuntimeStore
    connector,req,learning,runtime,remote,_,_,kwargs=setup
    now=time.time();durable(remote,req,now)
    connector.sync_grants(now=now)
    scope=dict(workspace_id=req.workspaceId,right="private_training",case_id=req.caseId,template_id=req.template.id,
        fields=["input.text"],model_bundle_id=req.modelBundleId,template_commitment=req.template_commitment())
    original=learning.check_right(**scope,now=now)
    for index in range(3):
        item=req.model_copy(deep=True);item.caseId=f"pause-case-{index}";item.sourceGroupId=item.caseId;item.input.text+=str(index)
        evaluate(learning,item)
        learning.add_feedback(workspace_id=item.workspaceId,evaluation_id=item.input_commitment(),input_commitment=item.input_commitment(),
            template_commitment=item.template_commitment(),annotator_id="human-1",labels={"tone":"suitable"},exposed_to_ai=False,independent_human=True)
    snapshot=learning.create_snapshot(req.workspaceId,req.template.id,req.template.version)
    learning.register_model_lineage("paused-candidate",snapshot["id"],req.workspaceId)
    remote["settings"]["mode"]=mode;remote["authorizationLease"]=None
    assert connector.sync_grants(now=now+1)["mode"]==mode
    with pytest.raises(PermissionError):learning.check_right(**scope,now=now+1)
    with pytest.raises(PermissionError,match="renewal"):learning.load_snapshot(snapshot["id"],req.workspaceId,now=now+1)
    with learning.transaction() as db:
        assert db["grants"][original[0]]["revoked_at"] is None
        assert db["snapshots"][snapshot["id"]].get("invalidated_at") is None
        assert db["lineage"]["paused-candidate"]["retired_at"] is None
        assert connector._state(db)["authorization_lease"] is None
    # Reopen the encrypted disk stores, not just the in-memory connector.
    learning=LearningStore(learning.root,learning.root.parent/"key")
    restarted=RateLoopConnector(**{**kwargs,"learning":learning,"runtime":RuntimeStore(runtime.path,learning.root.parent/"key")},metadata_upload_enabled=True)
    assert restarted.sync_grants(now=now+2)["mode"]==mode
    with pytest.raises(PermissionError):learning.check_right(**scope,now=now+2)
    remote["settings"]["mode"]="shadow"
    remote["authorizationLease"]={"leaseId":"resumed-lease","issuedAt":iso(now+3),"expiresAt":iso(now+803),
        "revocationWatermark":remote["revocationWatermark"],"workspaceId":req.workspaceId,"recipientApiKeyId":"api-key-1"}
    restarted.sync_grants(now=now+3)
    assert learning.check_right(**scope,now=now+3)==original
    assert learning.load_snapshot(snapshot["id"],req.workspaceId,now=now+3)["id"]==snapshot["id"]
    with learning.transaction() as db: assert db["lineage"]["paused-candidate"]["retired_at"] is None


@pytest.mark.parametrize("withdrawal",["revoked","missing","expired","local"])
def test_paused_sync_never_revives_real_withdrawal(setup,withdrawal):
    connector,req,learning,_,remote,_,_,_=setup
    now=time.time();durable(remote,req,now)
    if withdrawal=="expired":remote["consents"][0]["expiresAt"]=iso(now+10)
    connector.sync_grants(now=now)
    with learning.transaction() as db: local_id=connector._state(db)["consents"]["permission-1"]["local_id"]
    original=deepcopy(remote["consents"])
    remote["settings"]["mode"]="paused";remote["authorizationLease"]=None
    if withdrawal=="revoked":remote["consents"][0]["revokedAt"]=iso(now+1)
    elif withdrawal=="missing":remote["consents"]=[]
    elif withdrawal=="local":learning.revoke_grant(local_id,req.workspaceId,now=now+1)
    connector.sync_grants(now=now+20)
    with learning.transaction() as db: assert db["grants"][local_id]["revoked_at"] is not None
    remote["consents"]=original;remote["settings"]["mode"]="shadow"
    remote["authorizationLease"]={"leaseId":"resumed-lease","issuedAt":iso(now+21),"expiresAt":iso(now+821),
        "revocationWatermark":remote["revocationWatermark"],"workspaceId":req.workspaceId,"recipientApiKeyId":"api-key-1"}
    if withdrawal=="expired":connector.sync_grants(now=now+21)
    else:
        with pytest.raises((PermissionError,ValueError)):connector.sync_grants(now=now+21)
    with learning.transaction() as db: assert db["grants"][local_id]["revoked_at"] is not None


def test_initial_paused_consent_creates_no_grant_and_poisoned_mirror_requires_fresh_owner_consent(setup):
    connector,req,learning,_,remote,_,_,_=setup
    now=time.time();durable(remote,req,now);remote["settings"]["mode"]="paused";remote["authorizationLease"]=None
    with learning.transaction() as db: original_ids=set(db["grants"])
    connector.sync_grants(now=now)
    with learning.transaction() as db: assert set(db["grants"])==original_ids
    durable(remote,req,now+1);remote["settings"]["mode"]="shadow";connector.sync_grants(now=now+1)
    connector._revoke_mirrors("invalid_remote_grant_state")
    remote["settings"]["mode"]="paused";remote["authorizationLease"]=None
    assert connector.sync_grants(now=now+2)["mode"]=="paused"
    durable(remote,req,now+3);remote["settings"]["mode"]="shadow"
    with pytest.raises(PermissionError):connector.sync_grants(now=now+3)
    # A fresh explicit server consent gets a new mirror; the old one stays revoked.
    remote["consents"][0].update(consentId="fresh-owner-permission",revision=2)
    remote["revocationWatermark"]=2;remote["authorizationLease"]["revocationWatermark"]=2
    assert connector.sync_grants(now=now+3)["mirroredConsents"]==1
    with learning.transaction() as db:
        assert connector._state(db)["consents"]["fresh-owner-permission"]["local_id"] not in original_ids
        assert any(g["revoked_at"] is not None for g in db["grants"].values())


@pytest.mark.parametrize("mode",["off","paused"])
@pytest.mark.parametrize("stray_lease",[None,{"forged":"not-authority"}])
def test_inactive_mode_never_creates_authority_even_with_stray_lease(setup,mode,stray_lease):
    connector,req,learning,_,remote,_,_,_=setup
    now=time.time();durable(remote,req,now);connector.sync_grants(now=now)
    remote["settings"]["mode"]=mode;remote["authorizationLease"]=stray_lease
    connector.sync_grants(now=now+1)
    with pytest.raises(PermissionError):
        learning.check_right(workspace_id=req.workspaceId,right="private_training",case_id=req.caseId,template_id=req.template.id,
            fields=["input.text"],model_bundle_id=req.modelBundleId,template_commitment=req.template_commitment(),now=now+1)
    with learning.transaction() as db:assert connector._state(db)["authorization_lease"] is None


@pytest.mark.parametrize("revoked_purpose",["ai_use","private_learning"])
def test_revocation_during_pause_preserves_independent_durable_permission(setup,revoked_purpose):
    connector,req,learning,_,remote,_,_,_=setup
    now=time.time();durable(remote,req,now)
    ai=deepcopy(remote["consents"][0]);ai.update(consentId="ai-permission",purpose="ai_use")
    remote["consents"].append(ai);connector.sync_grants(now=now)
    with learning.transaction() as db: before=deepcopy(connector._state(db)["consents"])
    remote["settings"]["mode"]="paused";remote["authorizationLease"]=None
    for consent in remote["consents"]:
        if consent["purpose"]==revoked_purpose:consent["revokedAt"]=iso(now+1)
    connector.sync_grants(now=now+2)
    remote["settings"]["mode"]="shadow"
    remote["authorizationLease"]={"leaseId":"resumed-lease","issuedAt":iso(now+3),"expiresAt":iso(now+803),
        "revocationWatermark":remote["revocationWatermark"],"workspaceId":req.workspaceId,"recipientApiKeyId":"api-key-1"}
    connector.sync_grants(now=now+3)
    with learning.transaction() as db:
        for record in before.values():
            grant=db["grants"][record["local_id"]]
            if record["consent"]["purpose"]==revoked_purpose:assert grant["revoked_at"] is not None
            else:
                assert grant["revoked_at"] is None
                assert grant["authorization_until"]>now+3
