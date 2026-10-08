"""Opt-in process-local timing. No identifiers, inputs, labels or remote telemetry."""
from contextlib import contextmanager
from contextvars import ContextVar
import math
import time

_STAGES = frozenset({'load', 'prepare', 'infer', 'receive', 'deliver', 'total'})
_current = ContextVar('evaluator_timings', default=None)


def record_stages(values):
    target = _current.get()
    if target is None:
        return
    for key, value in values.items():
        if key not in _STAGES or type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid content-free timing stage')
        target[key] = target.get(key, 0) + value


@contextmanager
def capture_timings():
    # The wire lease contains no queue-entry timestamp. Unknown is not zero;
    # queue delay must be measured by the server that owns that timestamp.
    values = {'queue': None}
    token = _current.set(values)
    started = time.perf_counter()
    try:
        yield values
    finally:
        values['total'] = (time.perf_counter()-started)*1000
        _current.reset(token)


@contextmanager
def measure_stage(name):
    if name not in _STAGES:
        raise ValueError('Unknown timing stage')
    started = time.perf_counter()
    try:
        yield
    finally:
        record_stages({name: (time.perf_counter()-started)*1000})
