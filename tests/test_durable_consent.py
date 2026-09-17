from copy import deepcopy
import time

import pytest

from test_connector import setup, iso, evaluate, export_labels
from rateloop_evaluator.protocol import commitment, utc_now


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
