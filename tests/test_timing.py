import pytest

from rateloop_evaluator.timing import capture_timings, measure_stage, record_stages
from test_worker import website
from test_connector import setup


def test_timing_is_request_local_content_free_and_queue_is_unknown():
    with capture_timings() as first:
        record_stages({'prepare': 2, 'infer': 3})
        with capture_timings() as nested:
            record_stages({'infer': 9})
        assert first['infer'] == 3 and nested['infer'] == 9
        with pytest.raises(ValueError): record_stages({'input': 'private'})
        with pytest.raises(ValueError): record_stages({'infer': float('nan')})
    with capture_timings() as second:
        assert set(second) == {'queue'}
        with measure_stage('deliver'): pass
    assert first['queue'] is second['queue'] is None
    assert first['total'] >= 0 and second['deliver'] >= 0


def test_worker_records_real_stages_without_changing_receipt_or_retention(website):
    worker, _, _, _, _, requests = website
    assert worker.run_once()['state'] == 'completed'
    assert set(worker.last_timings) == {'queue', 'receive', 'load', 'prepare', 'infer', 'deliver', 'total'}
    assert worker.last_timings['queue'] is None
    assert all(value >= 0 for name, value in worker.last_timings.items() if name != 'queue')
    assert all(b'stages' not in request.content and b'timings' not in request.content for request in requests)
