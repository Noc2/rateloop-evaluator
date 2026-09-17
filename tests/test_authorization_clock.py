"""Remote clocks never extend local execution or alter durable consent lineage."""
from copy import deepcopy
import time

import pytest

from rateloop_evaluator.authorization import AUTHORIZATION_SECONDS, CLOCK_SKEW_SECONDS
from rateloop_evaluator.connector import AuthorizationLeaseRejected
from rateloop_evaluator.protocol import commitment
from test_connector import setup, iso
from test_durable_consent import durable
from test_worker import website


def scope(req):
    return dict(workspace_id=req.workspaceId,right="private_training",case_id=req.caseId,template_id=req.template.id,
        fields=["input.text"],model_bundle_id=req.modelBundleId,template_commitment=req.template_commitment())


@pytest.mark.parametrize("skew",[-5,0,.042,5])
def test_remote_issue_clock_and_local_learning_share_bounded_expiry_and_stable_wire_lineage(setup,skew):
    connector,req,learning,_,remote,_,_,_=setup
    now=float(int(time.time()));durable(remote,req,now)
    remote["consents"][0]["issuedAt"]=iso(now+skew)
    remote["authorizationLease"].update(issuedAt=iso(now+skew),expiresAt=iso(now+skew+AUTHORIZATION_SECONDS))
    wire=deepcopy(remote)
    connector.sync_grants(now=now)
    ids=learning.check_right(**scope(req),now=now)
    expected=min(now+skew+AUTHORIZATION_SECONDS-CLOCK_SKEW_SECONDS,now+AUTHORIZATION_SECONDS)
    assert expected<=now+AUTHORIZATION_SECONDS
    with learning.transaction() as database:
        grant=database["grants"][ids[0]]
        mirror=connector._state(database)["consents"][wire["consents"][0]["consentId"]]
        assert grant["authorization_until"]==pytest.approx(expected)
        assert mirror["consent"]==wire["consents"][0]
        assert mirror["digest"]==commitment(wire["consents"][0],"rateloop.durable-consent.v1")
        assert connector._state(database)["authorization_lease"]==wire["authorizationLease"]
        original_created=grant["created_at"]
    assert remote==wire
    assert learning.check_right(**scope(req),now=expected-.001)==ids
    with pytest.raises(PermissionError): learning.check_right(**scope(req),now=expected)
    later=now+86400
    remote["authorizationLease"].update(leaseId="renewed-next-day",issuedAt=iso(later+skew),expiresAt=iso(later+skew+900))
    connector.sync_grants(now=later)
    assert learning.check_right(**scope(req),now=later)==ids
    with learning.transaction() as database:
        assert database["grants"][ids[0]]["created_at"]==original_created
        assert database["grants"][ids[0]]["authorization_until"]<=later+900
        assert connector._state(database)["consents"][wire["consents"][0]["consentId"]]["digest"]==mirror["digest"]


def test_finite_consent_expiry_is_shortened_for_execution_but_not_rewritten(setup):
    connector,req,learning,_,remote,_,_,_=setup
    now=float(int(time.time()));durable(remote,req,now)
    remote["consents"][0]["expiresAt"]=iso(now+20)
    wire=deepcopy(remote["consents"][0]);connector.sync_grants(now=now)
    ids=learning.check_right(**scope(req),now=now+14.999)
    with learning.transaction() as database:
        grant=database["grants"][ids[0]]
        assert grant["expires_at"]==now+20
        assert grant["authorization_until"]==now+15
        assert connector._state(database)["consents"][wire["consentId"]]["consent"]==wire
    with pytest.raises(PermissionError): learning.check_right(**scope(req),now=now+15)


@pytest.mark.parametrize("remaining",[-1,0,4.999,5])
def test_expired_or_nearly_expired_remote_lease_cannot_authorize_processing(setup,remaining):
    connector,req,learning,_,remote,_,_,_=setup
    now=float(int(time.time()));durable(remote,req,now)
    remote["authorizationLease"].update(issuedAt=iso(now-895),expiresAt=iso(now+remaining))
    with pytest.raises(AuthorizationLeaseRejected) as rejected: connector.sync_grants(now=now)
    assert rejected.value.reason=="expired"
    with pytest.raises(PermissionError): learning.check_right(**scope(req),now=now)


def test_lease_just_above_conservative_expiry_boundary_still_expires_on_time(setup):
    connector,req,learning,_,remote,_,_,_=setup
    now=float(int(time.time()));durable(remote,req,now)
    remote["authorizationLease"].update(issuedAt=iso(now-894),expiresAt=iso(now+5.001))
    connector.sync_grants(now=now)
    learning.check_right(**scope(req),now=now)
    with pytest.raises(PermissionError): learning.check_right(**scope(req),now=now+.002)


@pytest.mark.parametrize("field",["issuedAt","revokedAt"])
def test_durable_consent_rejects_more_than_five_seconds_in_future(setup,field):
    connector,req,_,_,remote,_,_,_=setup
    now=float(int(time.time()));durable(remote,req,now)
    remote["consents"][0][field]=iso(now+5.001)
    with pytest.raises(ValueError,match="consent"): connector.sync_grants(now=now)


def test_observed_revocation_within_clock_skew_immediately_withdraws_rights(setup):
    connector,req,learning,_,remote,_,_,_=setup
    now=float(int(time.time()));durable(remote,req,now);connector.sync_grants(now=now)
    ids=learning.check_right(**scope(req),now=now)
    remote["consents"][0]["revokedAt"]=iso(now+5)
    connector.sync_grants(now=now)
    with pytest.raises(PermissionError): learning.check_right(**scope(req),now=now)
    with learning.transaction() as database:
        assert database["grants"][ids[0]]["revoked_at"] is not None


def test_finite_consent_near_expiry_is_not_mirrored_as_usable(setup):
    connector,req,learning,_,remote,_,_,_=setup
    now=float(int(time.time()));durable(remote,req,now)
    remote["consents"][0]["expiresAt"]=iso(now+5)
    connector.sync_grants(now=now)
    with pytest.raises(PermissionError): learning.check_right(**scope(req),now=now)


@pytest.mark.parametrize("duration",[0,-.001,900.001])
def test_authorization_duration_is_positive_and_never_above_900_seconds(setup,duration):
    connector,req,_,_,remote,_,_,_=setup
    now=float(int(time.time()));durable(remote,req,now)
    remote["authorizationLease"].update(issuedAt=iso(now),expiresAt=iso(now+duration))
    with pytest.raises(AuthorizationLeaseRejected) as rejected: connector.sync_grants(now=now)
    assert rejected.value.reason==("excessive_duration" if duration>900 else "invalid_duration")


def test_legacy_offline_grants_keep_original_strict_issue_boundary(setup):
    connector,_,_,_,remote,_,_,_=setup
    now=float(int(time.time()))
    remote["grants"][0]["issuedAt"]=iso(now+.042)
    with pytest.raises(ValueError,match="issue time"): connector.sync_grants(now=now)


def test_job_lease_uses_the_same_clock_tolerance_without_widening_existing_limit(website,monkeypatch):
    worker,req,_,_,_,_=website
    now=float(int(time.time()));monkeypatch.setattr("rateloop_evaluator.worker.time.time",lambda:now)
    job={"jobId":"test-job","modelBundleId":req.modelBundleId,"inputCommitment":req.input_commitment(),
        "templateCommitment":req.template_commitment(),"leaseToken":"x"*32,"leaseExpiresAt":iso(now+120+CLOCK_SKEW_SECONDS)}
    worker._validate_claim(job)
    job["leaseExpiresAt"]=iso(now+120+CLOCK_SKEW_SECONDS+.001)
    with pytest.raises(ValueError,match="120 seconds"): worker._validate_claim(job)
