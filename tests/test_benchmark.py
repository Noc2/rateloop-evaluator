import pytest
from rateloop_evaluator.benchmark import percentile, benchmark_backend


def test_percentiles_interpolate_and_do_not_claim_maximum_as_p95():
    assert percentile([30, 10, 20], .5) == 20
    assert percentile([0, 100], .95) == 95
    with pytest.raises(ValueError):
        percentile([], .5)


def test_benchmark_rejects_loaded_model_for_cold_start():
    class Loaded:
        _model = object()
    with pytest.raises(ValueError, match="fresh"):
        benchmark_backend(Loaded(), [{"text": "example", "questions": []}])
