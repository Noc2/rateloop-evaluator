from types import SimpleNamespace
from copy import deepcopy
import pytest

from rateloop_evaluator import worker_runtime


def test_worker_warms_before_returning_and_shares_identical_weights(monkeypatch):
    events=[]; validators={}
    records={name:{"artifact_root":"/model","manifest":{"id":name,"languages":[language],"files":{"weights":"digest"}}}
        for name,language in (("english","en"),("german","de"))}
    class Registry:
        def get(self,name,*args,**kwargs): return deepcopy(records[name])
        def serving_policy(self,name,_workspace,template,**kwargs): return {"bundle_id":name}
    class Backend:
        def __init__(self,path,device): events.append("construct")
        def load(self): events.append("load")
        def predict(self,text,questions): events.append("warm")
    def app(**kwargs):
        validators[kwargs["bundle"]["id"]]=kwargs["validate_bundle"]
        return SimpleNamespace(state=SimpleNamespace(evaluate=lambda request,_identity:kwargs["bundle"]["id"]))
    monkeypatch.setattr(worker_runtime,"GLiNERBackend",Backend)
    monkeypatch.setattr(worker_runtime,"create_app",app)
    connector=SimpleNamespace(workspace_id="workspace",learning=None,runtime=None)
    evaluate=worker_runtime.prepare_evaluator(connector,Registry(),["english","german"])
    assert events==["construct","load","warm","warm"]
    for name,language in (("english","en"),("german","de")):
        req=SimpleNamespace(modelBundleId=name,template=SimpleNamespace(language=language),template_commitment=lambda:"hash")
        assert validators[name](req)=={"bundle_id":name}
        assert evaluate(req)==name
    with pytest.raises(PermissionError): evaluate(SimpleNamespace(modelBundleId="other-workspace"))


def test_warmup_failure_never_returns_a_ready_worker(monkeypatch):
    class Backend:
        def __init__(self,*args): pass
        def load(self): raise RuntimeError("missing model")
    monkeypatch.setattr(worker_runtime,"GLiNERBackend",Backend)
    registry=SimpleNamespace(get=lambda *_:{"artifact_root":"/missing","manifest":{"files":{},"languages":["en"]}})
    with pytest.raises(RuntimeError,match="missing model"):
        worker_runtime.prepare_evaluator(SimpleNamespace(workspace_id="w"),registry,["bundle"])


def test_private_checkpoint_cache_is_bounded_and_routes_exact_authorized_queued_models(monkeypatch):
    resident=set(); events=[]; revoked=set()
    records={name:{"artifact_root":"/"+name,"manifest":{"id":name,"languages":["en"],
        "files":{"weights":name},**({"snapshot_id":"snapshot-"+name} if name!="base" else {})}}
        for name in ("base","candidate-a","candidate-b","candidate-c")}
    class Registry:
        def get(self,name,*args,**kwargs): return deepcopy(records[name])
        def serving_policy(self,name,*_args,**_kwargs):
            if name in revoked: raise PermissionError("Revoked source")
            return {"bundle_id":name}
    class Backend:
        def __init__(self,path,device): self.path=path
        def load(self):
            resident.add(self.path); events.append(("load",self.path))
            assert len(resident)<=1
        def unload(self): resident.discard(self.path); events.append(("unload",self.path))
        def predict(self,*_args): return self.path
    def app(**kw):
        def evaluate(request,_identity):
            kw["validate_bundle"](request)
            return kw["backend"].predict("Synthetic",[])
        return SimpleNamespace(state=SimpleNamespace(evaluate=evaluate))
    monkeypatch.setattr(worker_runtime,"GLiNERBackend",Backend)
    monkeypatch.setattr(worker_runtime,"create_app",app)
    evaluate=worker_runtime.prepare_evaluator(SimpleNamespace(workspace_id="workspace",learning=None,runtime=None),Registry(),list(records))
    assert resident=={"/candidate-c"}
    for name in ("candidate-a","candidate-b","candidate-c","base","candidate-a"):
        assert evaluate(SimpleNamespace(modelBundleId=name,template=None))=="/"+name
    assert resident=={"/candidate-a"}
    revoked.add("candidate-b"); previous=list(events)
    with pytest.raises(PermissionError): evaluate(SimpleNamespace(modelBundleId="candidate-b",template=None))
    assert events==previous
    evaluate.close()
    assert not resident
