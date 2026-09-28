"""Activation and serving presence must obey the same bundle-capacity boundary."""
import pytest

from rateloop_evaluator.presence import MAX_SERVED_MODEL_BUNDLES, WorkerPresence
from test_training_worker import runner, next_job
from test_cli import initialized


@pytest.mark.parametrize("base_count", [MAX_SERVED_MODEL_BUNDLES-1, MAX_SERVED_MODEL_BUNDLES])
def test_activation_checks_serving_capacity_before_server_acknowledgment(runner,base_count):
    worker,behavior,_,job,template,_,registry,_,_=runner
    assert worker.run_once()["state"]=="training_completed"
    worker.base_bundle_ids=["base",*[f"base-{index}" for index in range(1,base_count)]]
    WorkerPresence(worker.connector,worker.worker_id,worker.configured_bundles())
    next_job(behavior,job,"activate")
    result=worker.run_once()
    if base_count==MAX_SERVED_MODEL_BUNDLES:
        assert result["state"]=="training_failed"
        assert behavior["failed"][-1]["errorCode"]=="activation_capacity_reached"
        assert not any(result["action"]=="activate" for result in behavior["completed"])
        with pytest.raises(KeyError): registry.active("workspace-test",job["templateCommitment"],template.language)
        assert registry.serving_policy("base","workspace-test",template)["bundle_id"]=="base"
        assert len(worker.configured_bundles())==MAX_SERVED_MODEL_BUNDLES
    else:
        assert result["state"]=="training_completed"
        assert len(worker.configured_bundles())==MAX_SERVED_MODEL_BUNDLES
        WorkerPresence(worker.connector,worker.worker_id,worker.configured_bundles())
        with pytest.raises(ValueError,match="1-32"):
            WorkerPresence(worker.connector,worker.worker_id,worker.configured_bundles()+["one-too-many"])


def test_replacement_and_rollback_at_capacity_preserve_the_serving_invariant(runner):
    worker,behavior,_,job,template,_,registry,_,_=runner
    assert worker.run_once()["state"]=="training_completed"
    worker.base_bundle_ids=["base",*[f"base-{index}" for index in range(1,MAX_SERVED_MODEL_BUNDLES-1)]]
    next_job(behavior,job,"activate")
    assert worker.run_once()["state"]=="training_completed"
    next_job(behavior,job,"train","candidate-2")
    assert worker.run_once()["state"]=="training_completed"
    next_job(behavior,job,"activate","candidate-2")
    behavior["pending"]["jobId"]="activate-second-job"
    assert worker.run_once()["state"]=="training_completed"
    assert worker.configured_bundles()[-1]=="candidate-2"
    assert "candidate" not in worker.configured_bundles()
    WorkerPresence(worker.connector,worker.worker_id,worker.configured_bundles())
    next_job(behavior,job,"rollback","candidate")
    assert worker.run_once()["state"]=="training_completed"
    assert worker.configured_bundles()[-1]=="candidate"
    assert len(worker.configured_bundles())==MAX_SERVED_MODEL_BUNDLES
    assert registry.active("workspace-test",job["templateCommitment"],template.language)["bundle_id"]=="candidate"
    WorkerPresence(worker.connector,worker.worker_id,worker.configured_bundles())
