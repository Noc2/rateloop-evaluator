from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import httpx
import pytest

from rateloop_evaluator.connector import ConnectorUnavailable, RateLoopConnector
from rateloop_evaluator.learning import LearningStore, provision_key
from rateloop_evaluator.protocol import EvaluationRequest, commitment, make_result, utc_now
from rateloop_evaluator.storage import RuntimeStore


def iso(timestamp):
    return datetime.fromtimestamp(timestamp,timezone.utc).isoformat(timespec="milliseconds").replace("+00:00","Z")


def review_context():
    return {"policyId":"policy-1","policyVersion":1,"workflowKey":"reply","riskTier":"low", "audiencePolicyHash":"sha256:"+"a"*64,
            "metadataComplete":True,"execution":{"externalExecutionId":"execution-1","status":"completed","primarySpanId":"span-1",
            "generationSpans":[{"spanId":"span-1","role":"primary","provider":"local","requestedModel":"example-model"}]}}


@pytest.fixture
def setup(tmp_path):
    key=tmp_path/"key"; provision_key(key)
    learning=LearningStore(tmp_path/"learning",key)
    runtime=RuntimeStore(tmp_path/"runtime.sqlite",key)
    req=EvaluationRequest.model_validate_json((Path(__file__).parents[1]/"examples/reply-request.json").read_text())
    now=time.time()
    grant={"grantId":"server-grant-1","workspaceId":req.workspaceId,"apiKeyId":"api-key-1","purpose":"private_learning",
           "modelBundleId":req.modelBundleId,"templateCommitment":req.template_commitment(),"fields":["input","context","evidence","human_labels"],
           "publicWeightsAllowed":False,"issuedAt":iso(now-10),"expiresAt":iso(now+3600),"revokedAt":None,"revision":1}
    remote={"workspaceId":req.workspaceId,"recipientApiKeyId":"api-key-1","revocationWatermark":1,"settings":{"mode":"shadow"},"grants":[grant]}
    calls=[]
    behavior={"audit_status":200,"receipt_status":200,"grants_status":200,"labels":None,"selected":True}
    def transport(request):
        calls.append(request)
        assert request.headers["authorization"] == "Bearer secret-fixture-key"
        path=request.url.path
        if path.endswith("/grants"):
            return httpx.Response(behavior["grants_status"],json=remote)
        if path.endswith("/audits"):
            return httpx.Response(behavior["audit_status"],json={"auditId":"audit-1","kind":"random","selected":behavior["selected"],
                "aiExposed":False,"selectionProbabilityBps":1000,"blindingAssurance":"connector_attested","opportunityId":"human-opportunity-1"})
        if path.endswith("/receipts"):
            receipt=json.loads(request.content)
            return httpx.Response(behavior["receipt_status"],json={"schemaVersion":"rateloop.automated-eval-ingest-result.v2",
                "receiptId":"aev_"+"1"*40,"receiptHash":commitment(receipt,"rateloop.product-evaluator.v2"),"outcome":None if behavior.get("blind_receipt") else receipt["result"]["outcome"],
                "policy":{"mayReduceHumanReview":False},"replayed":False})
        if path.endswith("/labeled-data"):
            return httpx.Response(200,json=behavior["labels"])
        raise AssertionError(path)
    kwargs=dict(base_url="https://rateloop.example",api_key="secret-fixture-key",api_key_id="api-key-1",workspace_id=req.workspaceId,
                agent_id="agent-1",agent_version_id="version-1",learning=learning,runtime=runtime,transport=httpx.MockTransport(transport))
    connector=RateLoopConnector(**kwargs,metadata_upload_enabled=True)
    learning.add_grant(workspace_id=req.workspaceId,rights=["ai_use"],expires_at=now+7200,evidence="Independent local AI-use permission")
    return connector,req,learning,runtime,remote,behavior,calls,kwargs


def evaluate(learning,req):
    learning.record_evaluation(evaluation_id=req.input_commitment(),workspace_id=req.workspaceId,case_id=req.caseId,
        input_commitment=req.input_commitment(),template_commitment=req.template_commitment(),template=req.template.model_dump(),
        input_payload=req.input.model_dump(),model_bundle_id=req.modelBundleId,group_id=req.sourceGroupId)
    return make_result(workspaceId=req.workspaceId,caseId=req.caseId,modelBundleId=req.modelBundleId,
        inputCommitment=req.input_commitment(),templateCommitment=req.template_commitment(),outcome="uncertain",abstainReason="shadow_only",
        criteria=[{"questionId":q.id,"label":q.labels[0].id,"rawScores":{l.id:1/len(q.labels) for l in q.labels},
                   "probabilities":None,"calibrationId":None} for q in req.template.questions],durationMs=12,observedAt=utc_now())


def export_labels(connector,req,remote,*,outcome="positive"):
    with connector.learning.transaction() as database:
        state=connector._state(database)
        result=state["results"][req.input_commitment()]
        audit=state["audits"][req.input_commitment()]["response"]
    body={"schemaVersion":"rateloop.evaluator-labeled-data.v2","workspaceId":req.workspaceId,"grant":deepcopy(remote["grants"][0]),
          "revocationWatermark":remote["revocationWatermark"],"contentMode":"commitments_only","window":{"from":iso(time.time()-3600),"to":utc_now()},
          "items":[{"receiptId":"aev_"+"1"*40,"resultCommitment":result["resultCommitment"],"caseId":req.caseId,
             "inputCommitment":req.input_commitment(),"modelBundleId":req.modelBundleId,"templateCommitment":req.template_commitment(),
             "automatedOutcome":result["outcome"],"humanOutcome":outcome,"labelScope":"overall_human_verdict",
             "criterionTrainingRequiresAdjudication":True,"humanResultCommitment":"sha256:"+"b"*64,"responseCount":3,"observedAt":utc_now(),
             "audit":{"auditId":audit["auditId"],"kind":audit["kind"],"selectionProbabilityBps":audit["selectionProbabilityBps"],"aiExposed":False},"disagreement":False}],"truncated":False}
    return {**body,"exportDigest":commitment(body,"rateloop.product-evaluator.v2")}


def import_labels(connector,req):
    return connector.fetch_and_import_labels("server-grant-1",question_id="tone",template_commitment=req.template_commitment(),
                                            outcome_labels={"positive":"suitable","negative":"unsuitable"})


def test_destination_and_opt_in_boundaries(setup):
    connector,req,learning,_,_,_,calls,kwargs=setup
    for base in ("http://remote.example","https://user:pass@server.example","https://server.example/path","ftp://server.example"):
        with pytest.raises(ValueError):
            RateLoopConnector(**{**kwargs,"base_url":base})
    with pytest.raises(ValueError):
        RateLoopConnector(**{**kwargs,"base_url":"http://127.0.0.1"})
    local=RateLoopConnector(**{**kwargs,"base_url":"http://127.0.0.1","allow_insecure_loopback":True})
    local.close()
    disabled=RateLoopConnector(**kwargs)
    with pytest.raises(PermissionError):
        disabled.select_audit_before_scoring(req,review_context(),frozen_question_hash="sha256:"+"e"*64)
    with pytest.raises(PermissionError):
        disabled.flush()
    assert not calls
    assert connector.client.follow_redirects is False and connector.client._trust_env is False


def test_scoped_grant_mirror_expiry_and_independent_rights(setup):
    connector,req,learning,_,remote,_,_,_=setup
    assert connector.sync_grants()["mirroredGrants"]==1
    scope=dict(workspace_id=req.workspaceId,right="private_training",case_id=req.caseId,template_id=req.template.id,
               fields=["input.text"],model_bundle_id=req.modelBundleId,template_commitment=req.template_commitment())
    assert learning.check_right(**scope)
    for changes in ({"model_bundle_id":"other"},{"template_commitment":"sha256:"+"0"*64},{"right":"shared_contribution"},
                    {"right":"public_weight_distribution"},{"now":time.time()+7200}):
        with pytest.raises(PermissionError):
            learning.check_right(**{**scope,**changes})
    remote["settings"]["mode"]="paused"
    connector.sync_grants()
    # Pausing ratings is not a revocation of a separately granted learning purpose.
    assert learning.check_right(**scope)


def test_remote_revocation_and_rollback_fail_closed(setup):
    connector,req,learning,_,remote,_,_,_=setup
    connector.sync_grants()
    remote["grants"][0]["revokedAt"]=utc_now(); remote["grants"][0]["revision"]=2; remote["revocationWatermark"]=2
    assert connector.sync_grants()["mirroredGrants"]==0
    with pytest.raises(PermissionError):
        learning.check_right(workspace_id=req.workspaceId,right="private_training",case_id=req.caseId,template_id=req.template.id,
            fields=["input.text"],model_bundle_id=req.modelBundleId,template_commitment=req.template_commitment())
    remote["revocationWatermark"]=1; remote["grants"]=[]
    with pytest.raises(PermissionError,match="backwards"):
        connector.sync_grants()


def test_recipient_and_excessive_lease_rejected(setup):
    connector,_,_,_,remote,_,_,_=setup
    remote["recipientApiKeyId"]="another-key"
    with pytest.raises(PermissionError,match="recipient"):
        connector.sync_grants()
    remote["recipientApiKeyId"]="api-key-1"
    remote["grants"][0]["expiresAt"]=iso(time.time()+90000)
    with pytest.raises(ValueError,match="lease duration"):
        connector.sync_grants()


def test_shared_grant_never_implies_private_training(setup):
    connector,req,learning,_,remote,_,_,_=setup
    remote["grants"][0].update(purpose="shared_contribution",publicWeightsAllowed=True)
    connector.sync_grants()
    scope=dict(workspace_id=req.workspaceId,case_id=req.caseId,template_id=req.template.id,fields=["input.text"],
               model_bundle_id=req.modelBundleId,template_commitment=req.template_commitment())
    assert learning.check_right(**scope,right="shared_contribution")
    assert learning.check_right(**scope,right="public_weight_distribution")
    with pytest.raises(PermissionError):
        learning.check_right(**scope,right="private_training")


def test_durable_metadata_outbox_retries_without_raw_upload(setup):
    connector,req,learning,runtime,_,behavior,calls,kwargs=setup
    connector.sync_grants()
    result=evaluate(learning,req)
    receipt_id=connector.queue_result(result)
    assert connector.queue_result(result)==receipt_id
    behavior["receipt_status"]=503
    assert connector.flush()=={"delivered":0,"retrying":1,"rejected":0}
    with runtime.connect() as db:
        db.execute("UPDATE outbox SET next_attempt=0")
    behavior["receipt_status"]=200
    reopened=RateLoopConnector(**kwargs,metadata_upload_enabled=True)
    assert reopened.flush()=={"delivered":1,"retrying":0,"rejected":0}
    receipt_calls=[r for r in calls if r.url.path.endswith("/receipts")]
    assert len(receipt_calls)==2
    assert receipt_calls[0].headers["idempotency-key"]==receipt_calls[1].headers["idempotency-key"]
    assert req.input.text.encode() not in receipt_calls[0].content
    payload=json.loads(receipt_calls[0].content)
    assert set(payload)=={"schemaVersion","agentId","agentVersionId","result"}
    assert b"secret-fixture-key" not in runtime.path.read_bytes()


def test_redirect_does_not_forward_credentials_or_retry_to_other_origin(setup):
    _,req,learning,_,_,_,_,kwargs=setup
    calls=[]
    def redirect(request):
        calls.append(request)
        return httpx.Response(302,headers={"Location":"https://different.example/steal"})
    connector=RateLoopConnector(**{**kwargs,"transport":httpx.MockTransport(redirect)})
    with pytest.raises(ValueError,match="302"):
        connector.sync_grants()
    assert len(calls)==1


def test_blind_audit_precedes_scoring_and_selected_result_is_withheld(setup):
    connector,req,learning,_,remote,behavior,calls,_=setup
    connector.sync_grants()
    def local(request):
        assert calls[-1].url.path.endswith("/audits")
        return evaluate(learning,request)
    result=connector.run_with_audit(req,local,review_context(),frozen_question_hash="sha256:"+"e"*64)
    assert result["result"] is None and result["awaitingIndependentHuman"] is True
    behavior["labels"]=export_labels(connector,req,remote)
    imported=import_labels(connector,req)
    assert imported["imported"]==1 and not imported["rejected"]
    assert import_labels(connector,req)["duplicates"]==1
    assert connector.release_result(req.input_commitment())["caseId"]==req.caseId
    with pytest.raises(PermissionError,match="already been scored"):
        connector.select_audit_before_scoring(req,review_context(),frozen_question_hash="sha256:"+"e"*64)


def test_exposed_result_cannot_create_independent_label(setup):
    connector,req,learning,_,remote,behavior,_,_=setup
    connector.sync_grants()
    connector.run_with_audit(req,lambda r:evaluate(learning,r),review_context(),frozen_question_hash="sha256:"+"e"*64)
    connector.release_result(req.input_commitment())
    behavior["labels"]=export_labels(connector,req,remote)
    report=import_labels(connector,req)
    assert report["imported"]==0 and "blind" in report["rejected"][0]["reason"]


def test_offline_evaluation_never_claims_blinding(setup):
    connector,req,learning,_,_,behavior,_,_=setup
    connector.sync_grants(); behavior["audit_status"]=503
    with pytest.raises(ConnectorUnavailable):
        connector.run_with_audit(req,lambda r:evaluate(learning,r),review_context(),frozen_question_hash="sha256:"+"e"*64)
    report=connector.run_with_audit(req,lambda r:evaluate(learning,r),review_context(),frozen_question_hash="sha256:"+"e"*64,allow_offline=True)
    assert report["result"] is not None and report["blindingAssurance"]=="none" and not report["awaitingIndependentHuman"]
    with pytest.raises(PermissionError):
        connector.select_audit_before_scoring(req,review_context(),frozen_question_hash="sha256:"+"e"*64)


def test_offline_permission_cannot_override_a_known_pause(setup):
    connector,req,learning,_,remote,behavior,_,_=setup
    remote["settings"]["mode"]="paused"
    connector.sync_grants(); behavior["audit_status"]=503
    with pytest.raises(PermissionError,match="enabled workspace"):
        connector.run_with_audit(req,lambda r:evaluate(learning,r),review_context(),frozen_question_hash="sha256:"+"e"*64,allow_offline=True)
    with learning.transaction() as database:
        assert req.input_commitment() not in database["evaluations"]


def test_private_weights_distribution_does_not_grant_shared_learning(setup):
    connector,req,learning,_,remote,_,_,_=setup
    remote["grants"][0]["publicWeightsAllowed"]=True
    connector.sync_grants()
    scope = dict(workspace_id=req.workspaceId,case_id=req.caseId,template_id=req.template.id,
        model_bundle_id=req.modelBundleId,template_commitment=req.template_commitment(),fields=["input.text","human_labels"])
    learning.check_right(right="private_training",**scope)
    learning.check_right(right="public_weight_distribution",**scope)
    with pytest.raises(PermissionError): learning.check_right(right="shared_contribution",**scope)


def test_raw_human_context_is_rejected_before_any_upload(setup):
    connector,req,_,_,_,_,calls,_=setup
    context=review_context(); context["outputSummary"]="Secret customer content"
    with pytest.raises(ValueError,match="documented metadata"):
        connector.select_audit_before_scoring(req,context,frozen_question_hash="sha256:"+"e"*64)
    context=review_context(); context["execution"]["generationSpans"][0]["requestedModel"]="Secret customer content"
    with pytest.raises(ValueError,match="never free-form"):
        connector.select_audit_before_scoring(req,context,frozen_question_hash="sha256:"+"e"*64)
    assert calls==[]


def test_multicriterion_and_wrong_commitment_imports_are_rejected(setup):
    connector,req,learning,_,remote,behavior,_,_=setup
    connector.sync_grants()
    connector.run_with_audit(req,lambda r:evaluate(learning,r),review_context(),frozen_question_hash="sha256:"+"e"*64)
    behavior["labels"]=export_labels(connector,req,remote)
    behavior["labels"]["items"][0]["caseId"]="different-case"
    body={k:v for k,v in behavior["labels"].items() if k!="exportDigest"}
    behavior["labels"]["exportDigest"]=commitment(body,"rateloop.product-evaluator.v2")
    assert "exact local" in import_labels(connector,req)["rejected"][0]["reason"]
    behavior["labels"]=export_labels(connector,req,remote)
    with learning.transaction() as database:
        # Isolate import semantics; the complete protocol tests cover tampering.
        row=database["evaluations"][req.input_commitment()]
        row["template"]["questions"].append({**row["template"]["questions"][0],"id":"another-question"})
    report=import_labels(connector,req)
    assert report["imported"]==0 and "exactly one" in report["rejected"][0]["reason"]


def test_oversized_stream_is_bounded_before_download_completes(setup):
    *_,kwargs=setup
    consumed=[]
    class LargeStream(httpx.SyncByteStream):
        def __iter__(self):
            for i in range(400):
                consumed.append(i)
                yield b"x"*65536
    transport=httpx.MockTransport(lambda request:httpx.Response(200,stream=LargeStream()))
    connector=RateLoopConnector(**{**kwargs,"transport":transport})
    with pytest.raises(ValueError,match="bounded metadata limit"):
        connector.sync_grants()
    assert len(consumed)<200


def test_auth_failure_revokes_mirrored_training_permission(setup):
    connector,req,learning,_,_,behavior,_,_=setup
    connector.sync_grants(); behavior["grants_status"]=403
    with pytest.raises(PermissionError):
        connector.sync_grants()
    with pytest.raises(PermissionError):
        learning.check_right(workspace_id=req.workspaceId,right="private_training",case_id=req.caseId,template_id=req.template.id,
            fields=["input.text"],model_bundle_id=req.modelBundleId,template_commitment=req.template_commitment())


def test_deleted_case_cannot_reappear_in_connector_metadata(setup):
    connector,req,learning,_,_,_,_,_=setup
    connector.sync_grants()
    connector.run_with_audit(req,lambda r:evaluate(learning,r),review_context(),frozen_question_hash="sha256:"+"e"*64)
    old_result=connector.release_result(req.input_commitment())
    learning.delete_case(req.workspaceId,req.caseId)
    with pytest.raises(PermissionError,match="deleted"):
        connector.queue_result(old_result)
    with pytest.raises(PermissionError,match="deleted"):
        connector.select_audit_before_scoring(req,review_context(),frozen_question_hash="sha256:"+"e"*64)
    with pytest.raises(KeyError):
        connector.release_result(req.input_commitment())
