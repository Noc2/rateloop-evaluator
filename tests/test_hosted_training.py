"""Hosted coordination uses the same real encrypted grants/registry as local jobs."""
from copy import deepcopy
import multiprocessing
from threading import Event
from types import SimpleNamespace

import pytest

from rateloop_evaluator import hosted_training, training_worker
from rateloop_evaluator.connector import ConnectorUnavailable
from test_training_worker import runner, next_job
from test_cli import initialized


class ProcessContext:
    def __init__(self, operation, events, *, unavailable=False, alive=False, exitcode=0):
        self.operation,self.events,self.unavailable,self.alive=operation,events,unavailable,alive
        self.process=None
        self.exitcode=exitcode
    def Pipe(self, duplex): return multiprocessing.Pipe(duplex=duplex)
    def Process(self, *, target, args, **kwargs):
        assert target is hosted_training._training_process
        config, send=args
        owner=self
        class Process:
            exitcode=owner.exitcode
            def start(self):
                owner.events.append("child_start")
                # A duplicate endpoint mirrors the spawned child's own handle.
                connection=multiprocessing.connection.Connection(__import__('os').dup(send.fileno()))
                try:
                    connection.send({"poll":True})
                    connection.send({"unavailable":True} if owner.unavailable else {"result":owner.operation()})
                finally: connection.close()
            def is_alive(self): return owner.alive
            def join(self,timeout): owner.events.append("child_join")
            def terminate(self): owner.events.append("child_terminate"); owner.alive=False
            def kill(self): owner.events.append("child_kill"); owner.alive=False
            def close(self): owner.events.append("child_closed")
        self.process=Process()
        return self.process


def isolated(runner, **context_options):
    original,behavior,authorization,job,template,store,registry,changed,calls=runner
    events=[]; polls=[]
    context=ProcessContext(original.run_once,events,**context_options)
    config={"privateTraining":True,"stateDir":str(original.root)}
    worker=hosted_training.IsolatedHostedTrainingWorker(original.connector,registry,state_dir=original.root,
        worker_id=original.worker_id,model_dir=original.model_dir,model_bundle_ids=original.base_bundle_ids,
        hosted_config=config,release_inference=lambda:events.append("inference_released"),stop=Event(),
        process_context=context,on_poll=polls.append,
        on_models_changed=lambda ids:(events.append("allowlist_reloaded"),changed.append(ids)))
    return worker,events,polls


def test_hosted_training_requires_explicit_configuration(runner):
    original,*_=runner
    with pytest.raises(PermissionError,match="explicit"):
        hosted_training.IsolatedHostedTrainingWorker(original.connector,original.registry,state_dir=original.root,
            worker_id=original.worker_id,model_dir=original.model_dir,model_bundle_ids=original.base_bundle_ids,
            hosted_config={},release_inference=lambda:None,stop=Event())


def test_isolated_hosted_train_restart_activate_rollback_and_revoke(runner):
    worker,events,polls=isolated(runner)
    _,behavior,authorization,job,template,store,registry,changed,calls=runner
    assert worker.run_once()["state"]=="training_completed"
    assert events.index("inference_released")<events.index("child_start")<events.index("child_closed")<events.index("allowlist_reloaded")
    assert polls and all(polls)
    assert changed[-1]==["base"]
    next_job(behavior,job,"activate")
    assert worker.run_once()["state"]=="training_completed"
    assert changed[-1]==["base","candidate"]
    # A recreated coordinator reads the same signed candidate from the volume.
    restarted,_,_=isolated(runner)
    assert restarted.configured_bundles()==["base","candidate"]
    assert registry.serving_policy("candidate","workspace-test",template)["mode"]=="shadow"
    with pytest.raises((KeyError,PermissionError)):
        registry.serving_policy("candidate","another-workspace",template)
    behavior["permission"]=False
    restarted.sync_permissions()
    assert restarted.configured_bundles()==["base"]
    with pytest.raises(PermissionError): registry.serving_policy("candidate","workspace-test",template)


def test_child_unavailable_preserves_claim_and_releases_channels(runner):
    worker,events,_=isolated(runner,unavailable=True)
    with pytest.raises(ConnectorUnavailable): worker.run_once()
    assert worker._saved()["jobId"]=="training-job"
    assert "child_closed" in events and "allowlist_reloaded" not in events
    assert runner[1]["train_calls"]==0
    resumed,_,_=isolated(runner)
    assert resumed.run_once()["state"]=="training_completed"
    assert runner[1]["train_calls"]==1


def test_stop_terminates_child_without_inference_restart(runner):
    worker,events,_=isolated(runner,unavailable=True,alive=True)
    worker.stop.set()
    with pytest.raises(ConnectorUnavailable): worker.run_once()
    assert "child_terminate" in events and "child_closed" in events
    assert "allowlist_reloaded" not in events


def test_private_runtime_uses_spawn_not_fork(monkeypatch):
    # Importing the worker must not start processes or touch Torch. Construction
    # chooses spawn so inference model pages are not inherited into training.
    assert hosted_training.multiprocessing.get_context("spawn").get_start_method()=="spawn"


def test_hosted_time_limit_fails_durable_job_without_indefinite_retry(runner,monkeypatch):
    worker,events,_=isolated(runner,unavailable=True,alive=True)
    moments=iter((0,7200,7200))
    monkeypatch.setattr(hosted_training.time,'monotonic',lambda:next(moments,7200))
    assert worker.run_once()['state']=='training_failed'
    assert worker._saved() is None
    assert 'child_terminate' in events
    assert runner[1]['failed'][0]['errorCode']=='training_time_limit'


def test_parent_reconciles_committed_child_switch_when_result_message_was_lost(runner):
    worker,events,_=isolated(runner)
    original,behavior,_,job,template,store,registry,changed,calls=runner
    assert worker.run_once()['state']=='training_completed'
    # The child completes activation, but the parent never sees its result pipe.
    next_job(behavior,job,'activate')
    original.run_once()
    worker.sync_permissions()
    assert changed[-1]==['base','candidate']
    assert worker._serving_bundles==['base','candidate']
    # Fresh withdrawal also evicts the private ID on an otherwise idle poll.
    behavior['permission']=False
    worker.sync_permissions()
    assert changed[-1]==['base']


def test_abnormal_child_exit_fails_oversized_job_without_retry_loop(runner):
    worker,events,_=isolated(runner,unavailable=True,exitcode=-9)
    assert worker.run_once()['state']=='training_failed'
    assert worker._saved() is None
    assert runner[1]['failed'][0]['errorCode']=='training_resource_limit'


def test_daily_budget_survives_restart_and_crash_without_spawning(runner):
    worker,events,_=isolated(runner)
    # Reserving an operation models a crash before refund can be acknowledged.
    reservation=worker._reserve_budget(runner[3])
    assert reservation['seconds']==hosted_training.MAX_DAILY_TRAINING_SECONDS
    restarted,events,_=isolated(runner)
    assert restarted.run_once()['state']=='training_failed'
    assert runner[1]['failed'][0]['errorCode']=='training_daily_limit'
    assert 'child_start' not in events
    # Control-plane rollback is never blocked by the training resource budget.
    assert worker._reserve_budget({**runner[3],'action':'rollback'}) is None


def test_daily_budget_refunds_unused_time_and_resets_only_on_later_utc_day(runner,monkeypatch):
    worker,_,_=isolated(runner)
    now=hosted_training.time.time()
    reservation=worker._reserve_budget(runner[3])
    worker._refund_budget(reservation,100)
    with worker.connector.learning.transaction() as database:
        assert worker._state(database)['hosted_budget']['usedSeconds']==100
    monkeypatch.setattr(hosted_training.time,'time',lambda:now-86400)
    assert worker._reserve_budget(runner[3]) is False
    monkeypatch.setattr(hosted_training.time,'time',lambda:now+86400)
    assert worker._reserve_budget(runner[3])['seconds']==3600


def test_low_disk_refuses_training_before_claimed_content_is_loaded(runner,monkeypatch):
    worker,events,_=isolated(runner)
    monkeypatch.setattr(hosted_training.shutil,'disk_usage',lambda _:SimpleNamespace(free=100))
    assert worker.run_once()['state']=='training_failed'
    assert runner[1]['failed'][0]['errorCode']=='training_resource_limit'
    assert not runner[1]['completed'] and 'child_start' not in events
    assert not any(str(call.url.path).endswith('/content') for call in runner[-1])


def test_exhausted_budget_still_acknowledges_already_computed_result(runner):
    original,behavior,*_=runner
    behavior['offline_complete']=True
    with pytest.raises(ConnectorUnavailable): original.run_once()
    assert 'result' in original._saved()
    worker,events,_=isolated(runner)
    worker._reserve_budget(runner[3])
    behavior['offline_complete']=False
    assert worker.run_once()['state']=='training_completed'
    assert original._saved() is None
    assert 'child_start' not in events
    assert behavior['train_calls']==1
