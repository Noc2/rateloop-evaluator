"""Both queues preserve only bounded terminal failure intent until acknowledged."""
import json
import time

import httpx
import pytest

from rateloop_evaluator.connector import ConnectorUnavailable, _timestamp
from rateloop_evaluator.native_chat_pool import NativeChatPool
from test_connector import setup
from test_native_chat_pool import Backend, fixture
from test_worker import website


@pytest.mark.parametrize("consumer", ["native", "retained"])
@pytest.mark.parametrize("status,ack", [(200, {}), (200, {"retrying": True}), (200, {"retrying": 0}),
    (200, {"retrying": "false"}), (503, {})])
@pytest.mark.parametrize("expire", [False, True])
def test_terminal_failure_is_retried_without_reinference_or_lease_renewal(consumer, status, ack, expire, website, monkeypatch):
    now = [time.time()]
    monkeypatch.setattr("rateloop_evaluator.native_chat_pool.time.time", lambda: now[0])
    failures = []
    if consumer == "retained":
        worker, request, backend, _remote, behavior, calls = website
        behavior.update(receipt_status=422, fail_status=status, fail_body=ack)
        saved = worker._saved
        private_text = request.input.text
    else:
        job = fixture(); backend = Backend(); calls = []; behavior = {"fail_status": status, "fail_body": ack}
        private_text = job["content"]["request"]["input"]["text"]
        def transport(request):
            body = json.loads(request.content); action = body["action"]; calls.append(request)
            if action == "claim": return httpx.Response(200, json={"job": job})
            if action == "heartbeat_job": return httpx.Response(200, json={"leaseExpiresAt": job["leaseExpiresAt"]})
            if action == "complete": return httpx.Response(422, json={})
            if action == "fail":
                failures.append(body)
                return httpx.Response(behavior["fail_status"], json=behavior["fail_body"])
            raise AssertionError(action)
        worker = NativeChatPool(secret="s" * 32, base_url="https://www.rateloop.ai", bundles=[job["baseRegistration"]],
            backend=backend, transport=httpx.MockTransport(transport))
        saved = lambda: worker.pending[0] if worker.pending else None
    try:
        with pytest.raises(ConnectorUnavailable): worker.run_once()
        pending = saved()
        assert pending is not None and backend.calls == 1
        assert set(pending) == ({"jobId", "leaseToken", "leaseExpiresAt", "failureCode"}
            | ({"workspaceId"} if consumer == "native" else {"caseId"}))
        assert private_text not in json.dumps(pending)
        if consumer == "native": assert worker.pending[1] is None
        first_call_count = len(calls)
        if expire:
            now[0] = _timestamp(pending["leaseExpiresAt"])
        behavior.update(fail_status=200, fail_body={"retrying": False})
        if consumer == "retained":
            # A new process must recover failure intent from the encrypted store.
            worker = type(worker)(worker.connector, worker_id=worker.worker_id,
                model_bundle_ids=worker.model_bundle_ids, evaluate=worker.evaluate)
        result = worker.run_once()
        assert result["state"] == ("lease_lost" if expire else "failed")
        assert saved() is None and backend.calls == 1
        new_calls = calls[first_call_count:]
        actions = [json.loads(call.content).get("action") if consumer == "native" else call.url.path.rsplit("/", 1)[-1]
            for call in new_calls if consumer == "native" or "/jobs/" in call.url.path]
        assert actions == ([] if expire else ["fail"])
        if consumer == "retained":
            failures = [json.loads(call.content) for call in calls if call.url.path.endswith("/fail")]
        assert len(failures) == (1 if expire else 2)
        assert expire or failures[0] == failures[1]
    finally:
        if consumer == "native": worker.close()


def test_native_validation_failure_keeps_only_failure_intent_when_offline():
    job = fixture(); job["content"]["retainForTraining"] = True
    backend = Backend(); failures = []; actions = []
    def transport(request):
        body = json.loads(request.content); action = body["action"]; actions.append(action)
        if action == "claim": return httpx.Response(200, json={"job": job})
        if action == "heartbeat_job": return httpx.Response(200, json={"leaseExpiresAt": job["leaseExpiresAt"]})
        if action == "fail":
            failures.append(body)
            return httpx.Response(503 if len(failures) == 1 else 200, json={"retrying": False})
        raise AssertionError(action)
    pool = NativeChatPool(secret="s" * 32, base_url="https://www.rateloop.ai", bundles=[job["baseRegistration"]],
        backend=backend, transport=httpx.MockTransport(transport))
    try:
        with pytest.raises(ConnectorUnavailable): pool.run_once()
        assert pool.pending[1] is None and backend.calls == 0
        assert job["content"]["request"]["input"]["text"] not in json.dumps(pool.pending)
        assert pool.run_once() == {"state": "failed"}
        assert pool.pending is None and backend.calls == 0 and failures[0] == failures[1]
        assert actions == ["claim", "heartbeat_job", "fail", "fail"]
    finally: pool.close()


def test_retained_case_deletion_also_erases_pending_failure_intent(website):
    worker, request, _backend, _remote, behavior, _calls = website
    behavior.update(receipt_status=422, fail_status=503)
    with pytest.raises(ConnectorUnavailable): worker.run_once()
    assert worker._saved()["failureCode"] == "receipt_rejected_http_422"
    worker.connector.learning.delete_case(request.workspaceId, request.caseId)
    assert worker._saved() is None
