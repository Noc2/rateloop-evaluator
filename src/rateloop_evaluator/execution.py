"""Shared process/thread exclusion for one private state's model execution."""
from contextlib import contextmanager
import fcntl
from functools import wraps
import os
import threading


class ExecutionBusy(RuntimeError):
    pass


_guard=threading.Lock()
_locks={}
_held=threading.local()


@contextmanager
def model_execution(store):
    path=str((store.root/"model-execution.lock").absolute())
    with _guard:
        lock=_locks.setdefault(path,threading.RLock())
    if not lock.acquire(blocking=False):
        raise ExecutionBusy("Another local model operation is running; retry when it finishes")
    held=getattr(_held,"paths",set()); _held.paths=held
    descriptor=None
    try:
        if path in held:
            yield
            return
        descriptor=os.open(path,os.O_CREAT|os.O_RDWR|getattr(os,"O_NOFOLLOW",0),0o600)
        try: fcntl.flock(descriptor,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise ExecutionBusy("Another local model operation is running; retry when it finishes") from None
        held.add(path)
        try: yield
        finally: held.remove(path)
    finally:
        if descriptor is not None: os.close(descriptor)
        lock.release()


def serialized_training(function):
    @wraps(function)
    def wrapped(store,*args,**kwargs):
        with model_execution(store):
            return function(store,*args,**kwargs)
    return wrapped
