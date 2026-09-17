from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from rateloop_evaluator.execution import ExecutionBusy, model_execution
from rateloop_evaluator.training import train_snapshot
from test_connector import setup
from test_worker import website


def test_execution_lock_is_reentrant_and_excludes_another_thread(tmp_path):
    store=SimpleNamespace(root=tmp_path)
    outcomes=[]
    def contend():
        try:
            with model_execution(store): outcomes.append("entered")
        except ExecutionBusy: outcomes.append("busy")
    with model_execution(store):
        with model_execution(store):
            other=threading.Thread(target=contend); other.start();other.join()
    assert outcomes==["busy"]
    with model_execution(store): pass


def test_training_lock_prevents_worker_claim_and_optimizer_start_across_processes(website):
    worker,request,_,_,_,calls=website
    directory=worker.connector.learning.root
    source="""import fcntl,sys
f=open(sys.argv[1],'a');fcntl.flock(f,fcntl.LOCK_EX);print('locked',flush=True);sys.stdin.readline()
"""
    process=subprocess.Popen([sys.executable,"-c",source,str(directory/"model-execution.lock")],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    try:
        assert process.stdout.readline().strip()=="locked"
        assert worker.run_once()=={"state":"busy","reason":"local_model_operation"}
        assert not any("/jobs/" in request.url.path for request in calls)
        with pytest.raises(HTTPException) as busy:
            worker.evaluate(request)
        assert busy.value.status_code==429
        with pytest.raises(ExecutionBusy):
            train_snapshot(worker.connector.learning,"missing-snapshot","workspace",directory,directory/"output",bundle_id="bundle")
    finally:
        process.stdin.write("stop\n");process.stdin.flush();process.wait(timeout=5)
        process.stdin.close();process.stdout.close()
    assert worker.run_once()["state"]=="completed"
