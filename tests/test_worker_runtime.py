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
        def active(self,_workspace,_commitment,language,**kwargs): return {"bundle_id":{"en":"english","de":"german"}[language]}
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
