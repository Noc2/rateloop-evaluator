"""Warm, allowlisted inference shared by local and hosted outbound workers."""
from __future__ import annotations

from threading import RLock

from .backends import GLiNERBackend
from .service import Principal, create_app
from .templates import overall_approval
from .protocol import Template


class _PrivateCheckpointCache:
    """Keep one private checkpoint resident; signed IDs remain independently routable."""
    def __init__(self, device):
        self.device=device; self.key=None; self.backend=None; self.lock=RLock()

    def close(self):
        with self.lock:
            if self.backend is not None: self.backend.unload()
            self.backend=None; self.key=None

    def invoke(self, key, path, operation, *args):
        with self.lock:
            if key!=self.key:
                self.close()
                backend=GLiNERBackend(path,self.device)
                backend.load()
                self.backend=backend; self.key=key
            return getattr(self.backend,operation)(*args)


class _PrivateBackend:
    def __init__(self, cache, key, path): self.cache,self.key,self.path=cache,key,path
    def load(self): return self.cache.invoke(self.key,self.path,"load")
    def count_tokens(self,text,questions): return self.cache.invoke(self.key,self.path,"count_tokens",text,questions)
    def predict(self,text,questions): return self.cache.invoke(self.key,self.path,"predict",text,questions)


def prepare_evaluator(connector, registry, bundle_ids: list[str], *, device: str = "cpu"):
    """Validate and warm every bundle before any presence heartbeat is possible.

    Language-specific registrations share identical artifacts. Private models
    warm serially in a one-checkpoint cache and reload on demand for exact queued
    bundle IDs. No downloads, grants, customer content or training are involved.
    """
    workspace=connector.workspace_id
    identity=Principal(workspace,frozenset({"evaluate"}))
    apps={}; backends={}; private_cache=_PrivateCheckpointCache(device)
    for bundle_id in bundle_ids:
        record=registry.get(bundle_id,workspace)
        manifest=record["manifest"]
        # Registry verification binds the complete immutable artifact inventory.
        key=(record["artifact_root"],tuple(sorted(manifest["files"].items())))
        if key not in backends:
            backend=(_PrivateBackend(private_cache,key,record["artifact_root"]) if manifest.get("snapshot_id")
                else GLiNERBackend(record["artifact_root"],device))
            backend.load()
            backends[key]=backend
        backend=backends[key]
        for language in manifest["languages"]:
            template=Template.model_validate(manifest["template"]) if manifest.get("template") else overall_approval(language)
            backend.predict("Ready.",[question.model_dump() for question in template.questions])
        def validate(value, expected_id=bundle_id):
            return registry.serving_policy(expected_id,workspace,value.template,verify_artifacts=False)
        def allow_retention(value):
            with connector.learning.transaction() as database:
                return connector._state(database).get("collections",{}).get(value.input_commitment(),{}).get("trainingAllowed") is True
        apps[bundle_id]=create_app(backend=backend,bundle=manifest,learning=connector.learning,runtime=connector.runtime,
            tokens={"0"*64:identity},validate_bundle=validate,allow_training_retention=allow_retention)
    def evaluate(request):
        if request.modelBundleId not in apps: raise PermissionError("Unconfigured worker bundle")
        return apps[request.modelBundleId].state.evaluate(request,identity)
    def close():
        private_cache.close()
        for backend in backends.values():
            if not isinstance(backend,_PrivateBackend): backend.unload()
    evaluate.close=close
    return evaluate
