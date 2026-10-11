from copy import deepcopy
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from rateloop_evaluator import backends, worker_runtime
from rateloop_evaluator.model_identity import emit_loaded_model, loaded_model_identity
from test_backends import local_artifacts


def manifest():
    return {"files": {"tokenizer.json": "1" * 64, "model.safetensors": "2" * 64},
            "source": {"repository": backends.MODEL_ID, "revision": backends.MODEL_REVISION}}


def test_identity_is_canonical_complete_and_excludes_private_metadata(capsys):
    value = manifest()
    value["training"] = {"workspaceId": "private-workspace", "input": "private-prompt"}
    value["source"]["apiKey"] = "private-api-key"
    identity = loaded_model_identity(value)
    assert identity == {"schemaVersion": "rateloop.loaded-model.v1", "modelId": backends.MODEL_ID,
        "sourceRevision": backends.MODEL_REVISION, "trained": True, "weightsSha256": "2" * 64,
        "filesSha256": hashlib.sha256(json.dumps(value["files"], sort_keys=True,
            separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()}
    value["files"] = dict(reversed(list(value["files"].items())))
    assert loaded_model_identity(value) == identity
    emit_loaded_model(identity)
    out = capsys.readouterr().out
    assert json.loads(out) == {"event": "evaluator_model_loaded", **identity}
    assert "private" not in out and "tokenizer.json" not in out
    value["files"]["tokenizer.json"] = "3" * 64
    changed = loaded_model_identity(value)
    assert changed["weightsSha256"] == identity["weightsSha256"]
    assert changed["filesSha256"] != identity["filesSha256"]
    value["source"] = {"repository": "private-repo-secret", "revision": "private-revision"}
    assert loaded_model_identity(value)["modelId"] == "local"
    assert loaded_model_identity(value)["sourceRevision"] is None


@pytest.mark.parametrize("files", [{}, {"model.safetensors": "invalid"}, {"other": "1" * 64}])
def test_identity_requires_valid_verified_file_hashes(files):
    with pytest.raises(ValueError): loaded_model_identity({"files": files})


def test_actual_loader_and_cache_emit_only_after_success_and_on_every_reload(tmp_path, monkeypatch):
    local_artifacts(tmp_path)
    verified = backends.write_model_manifest(tmp_path, source={"repository": backends.MODEL_ID,
                                                             "revision": backends.MODEL_REVISION})
    events = []
    class Model:
        def to(self, _device): return self
        def float(self): return self
        def eval(self): events.append("loaded"); return self
    def load(path, *, local_files_only):
        assert path == str(tmp_path) and local_files_only is True
        return Model()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "gliner2", SimpleNamespace(AutoExtractor=SimpleNamespace(from_pretrained=load)))
    monkeypatch.setattr(backends, "release_checkpoint_file_cache", lambda *_: None)
    identities = []
    def observed(identity):
        assert events[-1] == "loaded"
        identities.append(identity)
    cache = worker_runtime._CheckpointCache("cpu", observed)
    key = (str(tmp_path), tuple(sorted(verified["files"].items())))
    cache.invoke(key, tmp_path, "load")
    cache.invoke(key, tmp_path, "load")
    assert identities == [loaded_model_identity(verified)]
    cache.close()
    cache.invoke(key, tmp_path, "load")
    assert identities == [loaded_model_identity(verified)] * 2
    cache.close()
    (tmp_path / "model.safetensors").write_text("tampered")
    with pytest.raises(ValueError, match="integrity"):
        cache.invoke(key, tmp_path, "load")
    assert len(identities) == 2 and cache.backend is None


@pytest.mark.parametrize("broken_stderr", [False, True])
def test_callback_failure_cannot_change_serving_or_disclose_exception(monkeypatch, capsys, broken_stderr):
    class Backend:
        manifest = manifest()
        def __init__(self, *_): pass
        def load(self): return "loaded-model"
        def predict(self): return "prediction"
    monkeypatch.setattr(worker_runtime, "GLiNERBackend", Backend)
    def fail(_): raise RuntimeError("private-api-key")
    if broken_stderr:
        class BrokenPipe:
            def write(self, *_): raise BrokenPipeError("private-log-pipe")
        monkeypatch.setattr(worker_runtime.sys, "stderr", BrokenPipe())
    cache = worker_runtime._CheckpointCache("cpu", fail)
    assert cache.invoke("key", "/private/path", "load") == "loaded-model"
    assert cache.invoke("key", "/private/path", "predict") == "prediction"
    out = capsys.readouterr()
    assert out.out == ""
    if broken_stderr: assert out.err == ""
    else: assert json.loads(out.err) == {"event": "evaluator_model_identity_unavailable"}


def test_failed_model_loading_never_reports_identity(monkeypatch, capsys):
    class Backend:
        manifest = manifest()
        def __init__(self, *_): pass
        def load(self): raise ValueError("cannot load")
    monkeypatch.setattr(worker_runtime, "GLiNERBackend", Backend)
    cache = worker_runtime._CheckpointCache("cpu", emit_loaded_model)
    with pytest.raises(ValueError, match="cannot load"):
        cache.invoke("key", "/private/path", "load")
    assert capsys.readouterr().out == ""


def test_shared_worker_warmup_and_native_reload_use_same_identity_projection(monkeypatch):
    identities = []
    base = manifest()
    candidate = deepcopy(base)
    candidate["files"]["model.safetensors"] = "4" * 64
    candidate["training"] = {"workspaceId": "private"}
    models = {"/base": base, "/candidate": candidate}
    class Backend:
        def __init__(self, path, _device): self.manifest = deepcopy(models[path])
        def load(self): return self
        def predict(self, *_): return "prediction"
        def unload(self): pass
    records = {name: {"artifact_root": "/" + name, "manifest": {"id": name, "languages": ["en"],
        "files": model["files"], **({"snapshot_id": "private"} if name == "candidate" else {"task_capability": "native"})}}
        for name, model in (("base", base), ("candidate", candidate))}
    registry = SimpleNamespace(get=lambda name, *_: records[name])
    monkeypatch.setattr(worker_runtime, "GLiNERBackend", Backend)
    monkeypatch.setattr(worker_runtime, "create_app", lambda **kw: SimpleNamespace(state=SimpleNamespace(
        evaluate=lambda *_: kw["backend"].predict("synthetic", []))))
    evaluate = worker_runtime.prepare_evaluator(SimpleNamespace(workspace_id="private", learning=None, runtime=None),
        registry, ["base", "candidate"], on_model_loaded=identities.append)
    evaluate.native_backend.load()
    evaluate(SimpleNamespace(modelBundleId="candidate"))
    evaluate.close()
    evaluate.native_backend.load()
    assert identities == [loaded_model_identity(model) for model in (base, candidate, base, candidate, base)]
