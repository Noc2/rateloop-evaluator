"""Reproducible local measurements; latency is not a quality guarantee."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.metadata
import math
import platform
import resource
import statistics
import threading
import time
from typing import Any


def percentile(values: list[float], probability: float) -> float:
    if not values or not 0 <= probability <= 1:
        raise ValueError("Percentile needs measurements and a probability in [0,1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def synchronize(device: str) -> None:
    if device == "mps":
        import torch
        torch.mps.synchronize()
    elif device == "cuda":
        import torch
        torch.cuda.synchronize()


@contextmanager
def measure_memory(device: str):
    import psutil
    process = psutil.Process()
    stats = {"rssStartBytes": process.memory_info().rss, "rssPeakSampledBytes": 0,
             "deviceAllocatedPeakSampledBytes": 0, "deviceDriverPeakSampledBytes": 0,
             "samplingIntervalMs": 10}
    stop = threading.Event()

    def sample() -> None:
        stats["rssPeakSampledBytes"] = max(stats["rssPeakSampledBytes"], process.memory_info().rss)
        if device in {"mps", "cuda"}:
            import torch
            if device == "mps":
                allocated = torch.mps.current_allocated_memory()
                driver = torch.mps.driver_allocated_memory()
            else:
                allocated = torch.cuda.memory_allocated()
                driver = torch.cuda.memory_reserved()
            stats["deviceAllocatedPeakSampledBytes"] = max(stats["deviceAllocatedPeakSampledBytes"], allocated)
            stats["deviceDriverPeakSampledBytes"] = max(stats["deviceDriverPeakSampledBytes"], driver)

    def poll() -> None:
        while not stop.wait(0.01):
            sample()

    sample()
    thread = threading.Thread(target=poll, daemon=True)
    thread.start()
    try:
        yield stats
    finally:
        stop.set()
        thread.join()
        sample()
        stats["rssEndBytes"] = process.memory_info().rss
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        stats["processLifetimePeakRssBytes"] = int(peak if platform.system() == "Darwin" else peak * 1024)


def environment(device: str) -> dict[str, Any]:
    import psutil
    versions = {}
    for name in ("rateloop-evaluator", "gliner2", "gliclass", "torch", "transformers", "peft", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "os": platform.system(), "osVersion": platform.release(),
        "architecture": platform.machine(), "processor": platform.processor(),
        "cpuCores": psutil.cpu_count(logical=False), "memoryBytes": psutil.virtual_memory().total,
        "device": device, "precision": "float32", "python": platform.python_version(),
        "packages": versions,
    }


def benchmark_backend(backend: Any, cases: list[dict[str, Any]], *,
                      iterations: int = 20, warmup: int = 3) -> dict[str, Any]:
    """Measure a fresh backend on supplied public/synthetic or authorized cases.

    Each case has text and questions; optional expected maps question IDs to
    human labels. Text never appears in the report. Run in a fresh process for
    cold-load and process-lifetime memory figures to have useful meaning.
    """
    if not cases or not 1 <= iterations <= 10000 or not 0 <= warmup <= 100:
        raise ValueError("Invalid benchmark cases or repetition count")
    if getattr(backend, "_model", None) is not None:
        raise ValueError("Cold-start benchmark requires a fresh backend")
    device = backend.device
    results = []
    with measure_memory(device) as memory:
        started = time.perf_counter()
        backend.predict(cases[0]["text"], cases[0]["questions"])
        synchronize(device)
        cold_ms = (time.perf_counter() - started) * 1000
        for case in cases:
            for _ in range(warmup):
                backend.predict(case["text"], case["questions"])
                synchronize(device)
            latencies = []
            scores = None
            for _ in range(iterations):
                synchronize(device)
                started = time.perf_counter()
                scores = backend.predict(case["text"], case["questions"])
                synchronize(device)
                latencies.append((time.perf_counter() - started) * 1000)
            result = {
                "questionCount": len(case["questions"]),
                "tokenCountIncludingSchema": backend.count_tokens(case["text"], case["questions"]),
                "iterations": iterations, "warmupIterations": warmup,
                "warmLatencyMs": {"p50": statistics.median(latencies), "p95": percentile(latencies, .95),
                                  "minimum": min(latencies), "maximum": max(latencies)},
            }
            if case.get("expected"):
                expected = case["expected"]
                if set(expected) != set(scores):
                    raise ValueError("Benchmark expected labels must cover every question")
                for question in case["questions"]:
                    if expected[question["id"]] not in {label["id"] for label in question["labels"]}:
                        raise ValueError("Benchmark label does not belong to its question")
                result["correctQuestions"] = sum(max(scores[qid], key=scores[qid].get) == label
                                                  for qid, label in expected.items())
                result["labeledQuestions"] = len(expected)
            results.append(result)
    return {
        "schemaVersion": "rateloop.benchmark.v1", "observedAt": datetime.now(timezone.utc).isoformat(),
        "environment": environment(device),
        "model": {key: value for key, value in (getattr(backend, "manifest", None) or {}).get("source", {}).items()
                  if key in {"repository", "revision", "library", "license", "baseWeightsSha256", "parentModelManifestSha256"}},
        "questionExecution": getattr(backend, "question_execution", "unspecified"),
        "coldStartAndFirstPredictionMs": cold_ms,
        "memory": memory, "cases": results,
        "limits": "Local sequential FP32 classification only; excludes network, queueing, calibration and human review. Sampled peaks may miss brief allocations. Fixture accuracy is not a production quality estimate.",
    }
