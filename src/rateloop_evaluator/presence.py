"""Explicit connected worker presence; never authority to use inputs or train."""
from __future__ import annotations

from contextlib import contextmanager
import secrets
import threading
import time

from .connector import ConnectorRejected, ConnectorUnavailable, RateLoopConnector, _opaque, _timestamp
from .execution import model_execution


class WorkerPresence:
    def __init__(self, connector: RateLoopConnector, worker_id: str, model_bundle_ids: list[str]):
        _opaque(worker_id)
        if not 1 <= len(model_bundle_ids) <= 32 or len(set(model_bundle_ids)) != len(model_bundle_ids):
            raise ValueError("Presence requires 1-32 explicit unique model bundles")
        for bundle in model_bundle_ids: _opaque(bundle)
        if not connector.metadata_upload_enabled:
            raise PermissionError("Worker presence requires explicit metadata upload permission")
        self.connector=connector; self.worker_id=worker_id; self.model_bundle_ids=list(model_bundle_ids)

    def report(self, state: str, *, operation_id: str | None = None) -> dict:
        if state not in ("ready","busy","training"):
            raise ValueError("Invalid worker presence state")
        if operation_id is not None: _opaque(operation_id)
        if state == "training" and operation_id is None:
            raise ValueError("Training presence requires an operation identity")
        if state == "busy" and operation_id is not None:
            raise ValueError("Busy presence cannot release a training operation")
        body={"workerId":self.worker_id,"modelBundleIds":self.model_bundle_ids,"state":state}
        if operation_id is not None: body["operationId"]=operation_id
        # This metadata-only update is idempotent for the same operation. Do not
        # broaden retries to permission failures, unknown 503s or connector writes.
        for attempt in range(3):
            try:
                response=self.connector._request("POST","/workers/heartbeat",json=body)
                break
            except ConnectorUnavailable as error:
                if not error.coordination_busy or attempt == 2: raise
                time.sleep((0.1,0.3)[attempt])
        if response.get("workerId") != self.worker_id or response.get("state") not in ("ready","busy","training"):
            raise ValueError("Presence response changed the worker identity or state")
        if operation_id is not None and response["state"] != state:
            raise ValueError("Presence response did not acknowledge this training operation")
        _timestamp(response.get("lastSeenAt"))
        return response


@contextmanager
def connected_training(connector: RateLoopConnector, *, worker_id: str, model_bundle_ids: list[str],
                       heartbeat_seconds: float = 30):
    """Hold execution and report one explicitly connected training operation.

    Local training without this context never contacts RateLoop. Training still
    checks the actual snapshot's permissions on every update; presence is not a
    training grant. A failed final status update expires on the server.
    """
    if not 0 < heartbeat_seconds <= 30:
        raise ValueError("Training presence refresh must be at most 30 seconds")
    presence=WorkerPresence(connector,worker_id,model_bundle_ids)
    with model_execution(connector.learning):
        if connector.sync_grants()["mode"] != "shadow":
            raise PermissionError("Connected training is paused")
        operation_id="training-"+secrets.token_hex(16)
        presence.report("training",operation_id=operation_id)
        stopped=threading.Event(); failures=[]
        def check():
            if failures: raise failures[0]
        def renew():
            while not stopped.wait(heartbeat_seconds):
                try:
                    if connector.sync_grants()["mode"] != "shadow":
                        raise PermissionError("Connected training is paused")
                    presence.report("training",operation_id=operation_id)
                except (ConnectorUnavailable,ConnectorRejected,PermissionError,ValueError) as error:
                    failures.append(error)
                    # The optimizer's original short authorization lease still
                    # bounds unavailable refreshes; explicit revocation is mirrored.
                    if not isinstance(error,ConnectorUnavailable): return
        thread=threading.Thread(target=renew,name="evaluator-training-presence",daemon=True)
        thread.start()
        try:
            yield check
            check()
        finally:
            stopped.set();thread.join(timeout=35)
            # Only this operation can release its fresh training status. A
            # concurrent worker's ordinary heartbeat cannot clear it.
            try: presence.report("ready",operation_id=operation_id)
            except (ConnectorUnavailable,ConnectorRejected,PermissionError,ValueError): pass
