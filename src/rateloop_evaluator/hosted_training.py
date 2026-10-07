"""Bounded, process-isolated training on an explicitly enabled private volume.

Only the authenticated website queue supplies jobs. The child inherits no model
objects, accepts no uploaded code or paths, and returns operational metadata.
Inference resumes after process exit, which releases optimizer/model memory.
"""
from __future__ import annotations

from argparse import Namespace
from contextlib import redirect_stderr, redirect_stdout
import multiprocessing
import os
import shutil
import time

from . import cli
from .connector import ConnectorUnavailable, RateLoopConnector
from .storage import RuntimeStore
from .training_worker import TrainingWorker


MAX_OPERATION_SECONDS = 3600
MAX_DAILY_TRAINING_SECONDS = 3600
MIN_TRAINING_FREE_BYTES = 3 * 1024**3


def _training_process(config, channel):
    connector = None
    try:
        # Upstream trainers can print samples, paths or tensor diagnostics. Never
        # send that output to hosted logs or the parent metadata channel.
        with open(os.devnull, "w") as sink, redirect_stdout(sink), redirect_stderr(sink):
            from .backends import offline_environment
            offline_environment()
            import torch
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            root, local, store, registry = cli.state(Namespace(state_dir=config["stateDir"]))
            c = config["connection"]
            connector = RateLoopConnector(base_url=c["baseUrl"], api_key=c["apiKey"], api_key_id=c["apiKeyId"],
                agent_id=c["agentId"], agent_version_id=c["agentVersionId"], workspace_id=config["workspaceId"],
                learning=store, runtime=RuntimeStore(root/"runtime.sqlite", local["encryptionKey"]),
                metadata_upload_enabled=True)
            worker = TrainingWorker(connector, registry, state_dir=root, worker_id=config["workerId"],
                model_dir=config["modelDir"], model_bundle_ids=[b["modelBundleId"] for b in config["bundles"]],
                device="cpu", on_poll=lambda healthy: channel.send({"poll": healthy}))
            result = worker.run_once()
            channel.send({"result": result})
    except Exception:
        # The durable job and fencing lease remain available for safe recovery.
        # Error strings may contain private content and never cross the channel.
        try: channel.send({"unavailable": True})
        except (BrokenPipeError, EOFError, OSError): pass
    finally:
        if connector is not None: connector.close()
        channel.close()


class IsolatedHostedTrainingWorker(TrainingWorker):
    """Claim with the parent, execute after unloading inference in a fresh child."""
    def __init__(self, *args, hosted_config, release_inference, stop, process_context=None,
                 max_operation_seconds=MAX_OPERATION_SECONDS, **kwargs):
        super().__init__(*args, **kwargs)
        if hosted_config.get("privateTraining") is not True:
            raise PermissionError("Hosted private training requires explicit configuration")
        if not 1 <= max_operation_seconds <= MAX_OPERATION_SECONDS:
            raise ValueError("Hosted operation duration exceeds its execution budget")
        self.hosted_config = hosted_config
        self.release_inference = release_inference
        self.stop = stop
        self.process_context = process_context or multiprocessing.get_context("spawn")
        self.max_operation_seconds = max_operation_seconds
        self._serving_bundles = None

    def sync_permissions(self, state="ready"):
        response=super().sync_permissions(state)
        current=self.configured_bundles()
        if self._serving_bundles is not None and current != self._serving_bundles:
            # Reconcile even if the child completed a switch immediately before
            # interruption prevented its result message reaching the parent.
            self.on_models_changed(current)
        self._serving_bundles=current
        return response

    def _reserve_budget(self, job, *, maximum_seconds=None):
        if job["action"] not in ("train","compare"): return None
        if maximum_seconds is None: maximum_seconds=self.max_operation_seconds
        if type(maximum_seconds) not in (int,float) or not 1<=maximum_seconds<=self.max_operation_seconds:
            raise ValueError("Invalid operation reservation duration")
        day=int(time.time()//86400)
        with self.connector.learning.transaction() as database:
            state=self._state(database)
            budget=state.setdefault("hosted_budget",{"utcDay":day,"usedSeconds":0.0})
            if budget["utcDay"] < day:
                budget.update(utcDay=day,usedSeconds=0.0)
            remaining=MAX_DAILY_TRAINING_SECONDS-budget["usedSeconds"] if budget["utcDay"]==day else 0
            if remaining < 30: return False
            reserved=min(maximum_seconds,remaining)
            # Charge before spawn. A process/machine crash cannot reset spend;
            # only this surviving coordinator refunds unused reserved seconds.
            budget["usedSeconds"]+=reserved
            return {"utcDay":day,"seconds":reserved}

    def _refund_budget(self, reservation, elapsed):
        if not reservation: return
        refund=max(0,reservation["seconds"]-max(0,elapsed))
        with self.connector.learning.transaction() as database:
            budget=self._state(database)["hosted_budget"]
            if budget["utcDay"]==reservation["utcDay"]:
                budget["usedSeconds"]=max(0,budget["usedSeconds"]-refund)

    def _resource_failure(self, job, code):
        saved=self._saved() or job
        if "result" in saved:
            # A child may persist completion while the parent is observing its
            # exit. Preserve the receipt/switch intent even after a late kill.
            return super().execute_pending(saved)
        saved["failureCode"]=code
        self._save(saved)
        return self._fail(saved)

    def execute_pending(self, job):
        if "result" in job:
            # A completed calculation needs only receipt/switch reconciliation.
            # A crash-reserved daily budget must never block that cleanup.
            return super().execute_pending(job)
        if job["action"]=="train" and shutil.disk_usage(self.root).free < MIN_TRAINING_FREE_BYTES:
            return self._resource_failure(job,"training_resource_limit")
        reservation=self._reserve_budget(job)
        if reservation is False: return self._resource_failure(job,"training_daily_limit")
        self.release_inference()
        receive, send = self.process_context.Pipe(duplex=False)
        process = self.process_context.Process(target=_training_process, args=(self.hosted_config, send),
            name="private-training", daemon=False)
        began=time.monotonic()
        allowance=reservation["seconds"] if reservation else self.max_operation_seconds
        deadline=began+allowance
        result = None
        started = False
        timed_out = False
        resource_failed = False
        try:
            process.start(); started = True; send.close()
            while not self.stop.is_set() and time.monotonic() < deadline:
                if receive.poll(1):
                    try: message = receive.recv()
                    except EOFError: break
                    if message == {"poll": True}:
                        self.on_poll(True)
                    elif isinstance(message, dict) and set(message) == {"result"}:
                        result = message["result"]
                    else:
                        break
                if not process.is_alive() and not receive.poll(): break
            # Result delivery precedes child cleanup; wait briefly for that to
            # finish before a new model can be loaded into the container.
            process.join(timeout=5)
            timed_out = time.monotonic() >= deadline and not self.stop.is_set()
            resource_failed = process.exitcode not in (None,0) and not self.stop.is_set()
            if not timed_out and not resource_failed and (process.is_alive() or process.exitcode != 0 or result is None or self.stop.is_set()):
                raise ConnectorUnavailable("Hosted training interrupted; the saved job will be reconciled")
        finally:
            if started and process.is_alive():
                process.terminate(); process.join(timeout=5)
                if process.is_alive(): process.kill(); process.join(timeout=5)
            receive.close(); send.close()
            if started: process.close()
            self._refund_budget(reservation,time.monotonic()-began)
        if timed_out or resource_failed:
            return self._resource_failure(job,"training_time_limit" if timed_out else "training_resource_limit")
        # The child updates the encrypted registry atomically. Revalidate grants
        # and rebuild the exact serving allowlist; never fall back to another ID.
        previous=self._serving_bundles
        self.sync_permissions()
        if self._serving_bundles == previous:
            self.on_models_changed(self.configured_bundles())
        return result
