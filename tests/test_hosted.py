from copy import deepcopy
import json
from pathlib import Path
import signal
import socket
import stat
import sys
from types import SimpleNamespace

import httpx
import pytest

from rateloop_evaluator import cli, hosted
from rateloop_evaluator.worker import single_worker
from test_cli import model_files


@pytest.fixture
def config(tmp_path):
    return {"schemaVersion":"rateloop.hosted-worker.v1","workspaceId":"workspace-test","workerId":"hosted-test",
        "modelDir":str(model_files(tmp_path)),"stateDir":str(tmp_path/"state"),
        "bundles":[{"language":"en","modelBundleId":"hosted-en"},{"language":"de","modelBundleId":"hosted-de"}],
        "connection":{"baseUrl":"https://rateloop.example","apiKey":"private-test-api-key","apiKeyId":"api-1",
            "agentId":"agent-1","agentVersionId":"version-1","metadataUploadEnabled":True},"pollSeconds":5,"healthPort":8080}


def test_bootstrap_restart_preserves_keys_bundle_metadata_and_no_permissions(config):
    first=hosted.bootstrap(config)
    root=Path(config["stateDir"])
    before={p.name:p.read_bytes() for p in (root/"encryption.key",root/"signing.key")}
    exports={p:Path(p).read_bytes() for p in first["registrations"]}
    assert hosted.bootstrap(config)==first
    assert all((root/name).read_bytes()==value for name,value in before.items())
    assert all(Path(p).read_bytes()==value for p,value in exports.items())
    _,_,store,_=cli.state(SimpleNamespace(state_dir=root))
    with store.transaction() as db: assert not db["grants"]
    for p in (root/"config.json",root/"connector.json",root/"encryption.key",root/"signing.key"):
        assert stat.S_IMODE(p.stat().st_mode)==0o600
    for path in first["registrations"]:
        export=json.loads(Path(path).read_text())
        assert export["scoreCapability"]["scoreType"]=="mutually_exclusive_softmax"
        assert export["quantization"]=="fp32"
        assert "private-test-api-key" not in Path(path).read_text()
    assert "private-test-api-key" not in json.dumps(first)


@pytest.mark.parametrize("field",["workspaceId","workerId","agentId","modelBundleId"])
def test_restart_refuses_rebinding_existing_volume(config,field):
    hosted.bootstrap(config)
    changed=deepcopy(config)
    if field=="agentId": changed["connection"][field]="another-agent"
    elif field=="modelBundleId": changed["bundles"][0][field]="another-bundle"
    else: changed[field]="another-identity"
    with pytest.raises(PermissionError): hosted.bootstrap(changed)


def test_configuration_is_private_strict_and_scoped(config,tmp_path):
    path=tmp_path/"hosted.json"; cli.write_private(path,config)
    assert hosted.read_config(path)==config
    for mutate in (lambda v:v.update(extra="unknown"),lambda v:v["bundles"].append(v["bundles"][0]),
        lambda v:v.update(healthPort=80),lambda v:v["connection"].update(metadataUploadEnabled=False)):
        changed=deepcopy(config); mutate(changed); cli.write_private(path,changed)
        with pytest.raises(ValueError): hosted.read_config(path)
    cli.write_private(path,config); path.chmod(0o644)
    with pytest.raises(ValueError): hosted.read_config(path)


def test_prepare_does_not_download_existing_model_and_runtime_never_provisions(config,monkeypatch):
    monkeypatch.setattr(hosted,"provision_model",lambda *_args,**_kwargs:pytest.fail("implicit download"))
    hosted.prepare_model(config)
    hosted.bootstrap(config)
    Path(config["modelDir"],"model.safetensors").write_text("tampered")
    with pytest.raises(ValueError): hosted.prepare_model(config)
    with pytest.raises(ValueError): hosted.bootstrap(config)


def test_health_requires_warm_fresh_success_and_rejects_inference():
    clock=[0.0]; health=hosted.WorkerHealth(clock=lambda:clock[0])
    assert health.response()[0]==503
    health.polled(True); assert health.response()[0]==503
    health.warm=True; assert health.response()==(200,{"status":"ready"})
    clock[0]=151; assert health.response()[0]==503
    health.polled(True); health.polled(False); assert health.response()[0]==503
    health.polled(True)
    with hosted.health_server(health,0,host="127.0.0.1") as port:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}",trust_env=False) as client:
            assert client.get("/healthz").json()=={"status":"ready"}
            assert client.get("/healthz").headers["cache-control"]=="no-store"
            assert client.get("/v1/evaluate").status_code==404
            assert client.post("/v1/evaluate",json={"private":"data"}).status_code==501
    assert health.response()[0]==503
    with socket.socket() as sock: assert sock.connect_ex(("127.0.0.1",port))!=0


def test_runtime_shutdown_restores_signals_releases_lock_and_closes_connector(config,monkeypatch):
    events=[]; health=hosted.WorkerHealth(); signals={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(set_num_threads=lambda n:None,set_num_interop_threads=lambda n:None))
    monkeypatch.setattr(hosted,"health_server",lambda _health,_port:__import__("contextlib").nullcontext())
    class Connector:
        def __init__(self,**kwargs): events.append("connected")
        def close(self): events.append("closed")
    class Worker:
        def __init__(self,*args,**kwargs): events.append("worker")
        def run(self):
            assert events==["connected","warm","worker"]
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM,None)
            assert self.stop.is_set()
    monkeypatch.setattr(hosted,"RateLoopConnector",Connector)
    monkeypatch.setattr(hosted,"prepare_evaluator",lambda *_args,**_kwargs:events.append("warm"))
    monkeypatch.setattr(hosted,"OutboundWorker",Worker)
    hosted.run_hosted(config)
    assert events[-1]=="closed"
    assert all(signal.getsignal(s)==handler for s,handler in signals.items())
    with single_worker(Path(config["stateDir"]),config["workerId"]): pass


def test_cli_errors_do_not_disclose_secret_config(config,tmp_path,capsys,monkeypatch):
    path=tmp_path/"hosted.json"; cli.write_private(path,config)
    def fail(_): raise ValueError("private-test-api-key")
    monkeypatch.setattr(hosted,"bootstrap",fail)
    assert hosted.main(["bootstrap","--config",str(path)])==1
    out=capsys.readouterr()
    assert "private-test-api-key" not in out.out+out.err
