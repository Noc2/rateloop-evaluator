"""Private, single-workspace hosted bootstrap and metadata-only health service."""
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import signal
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import cli
from .backends import MODEL_ID, MODEL_REVISION, provision_model, validate_local_model, offline_environment
from .connector import RateLoopConnector, _opaque
from .learning import provision_key, read_secret
from .protocol import EvaluationRequest
from .registry import provision_signing_key
from .storage import RuntimeStore
from .templates import overall_approval
from .worker import OutboundWorker, single_worker
from .worker_runtime import prepare_evaluator


def read_config(path: str | Path) -> dict:
    value=json.loads(read_secret(path))
    fields={"schemaVersion","workspaceId","workerId","modelDir","stateDir","bundles","connection","pollSeconds","healthPort"}
    if not isinstance(value,dict) or set(value)!=fields or value["schemaVersion"]!="rateloop.hosted-worker.v1":
        raise ValueError("Invalid hosted worker configuration")
    _opaque(value["workspaceId"]); _opaque(value["workerId"])
    for key in ("modelDir","stateDir"):
        path=Path(value[key])
        if not path.is_absolute() or path.is_symlink() or path.resolve()!=path:
            raise ValueError("Hosted paths must be absolute local non-symlink paths")
    if Path(value["modelDir"])==Path(value["stateDir"]) or Path(value["modelDir"]).is_relative_to(Path(value["stateDir"])):
        raise ValueError("Model and private state directories must be separate")
    if type(value["pollSeconds"]) not in (int,float) or not 1<=value["pollSeconds"]<=30:
        raise ValueError("Hosted polling must be 1-30 seconds")
    if type(value["healthPort"]) is not int or not 1024<=value["healthPort"]<=65535:
        raise ValueError("Invalid hosted health port")
    bundles=value["bundles"]
    if not isinstance(bundles,list) or not 1<=len(bundles)<=2:
        raise ValueError("Hosted worker requires one or two language bundles")
    for bundle in bundles:
        if not isinstance(bundle,dict) or set(bundle)!={"language","modelBundleId"} or bundle["language"] not in ("en","de"):
            raise ValueError("Invalid hosted language bundle")
        _opaque(bundle["modelBundleId"])
    if len({b["language"] for b in bundles})!=len(bundles) or len({b["modelBundleId"] for b in bundles})!=len(bundles):
        raise ValueError("Hosted bundle identities and languages must be unique")
    connection=value["connection"]
    if not isinstance(connection,dict) or set(connection)!={"baseUrl","apiKey","apiKeyId","agentId","agentVersionId","metadataUploadEnabled"}:
        raise ValueError("Invalid hosted connector configuration")
    if connection["metadataUploadEnabled"] is not True:
        raise ValueError("Hosted worker requires explicit metadata upload permission")
    return value


def prepare_model(config: dict) -> None:
    """Explicit provisioning process; never called by bootstrap or runtime."""
    destination=Path(config["modelDir"])
    if destination.exists():
        validate_pinned_model(destination)
        return
    destination.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    staging=Path(tempfile.mkdtemp(prefix=".provision-",dir=destination.parent))
    try:
        provision_model(staging,revision=MODEL_REVISION)
        validate_pinned_model(staging)
        staging.rename(destination)
    finally:
        if staging.exists(): shutil.rmtree(staging)


def validate_pinned_model(path: Path) -> dict:
    model=validate_local_model(path)
    if (model["source"].get("repository"),model["source"].get("revision"))!=(MODEL_ID,MODEL_REVISION) or model.get("training"):
        raise ValueError("Hosted bootstrap requires the reviewed immutable public model")
    return model


def bootstrap(config: dict) -> dict:
    """Preserve restart identity and keys, export metadata; never grant AI use."""
    validate_pinned_model(Path(config["modelDir"]))
    root=Path(config["stateDir"])
    if not root.exists():
        root.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        staging=Path(tempfile.mkdtemp(prefix=".state-bootstrap-",dir=root.parent))
        try:
            provision_key(staging/"encryption.key"); provision_signing_key(staging/"signing.key")
            cli.write_private(staging/"config.json",{"workspaceId":config["workspaceId"],
                "encryptionKey":str(root/"encryption.key"),"signingKey":str(root/"signing.key"),"tokens":[]})
            staging.rename(root)
        finally:
            if staging.exists(): shutil.rmtree(staging)
    args=Namespace(state_dir=str(root))
    _,local,store,registry=cli.state(args)
    if local["workspaceId"]!=config["workspaceId"]:
        raise PermissionError("Hosted volume belongs to another workspace")
    identity={key:config[key] for key in ("workspaceId","workerId","modelDir","bundles")}
    identity.update({key:config["connection"][key] for key in ("baseUrl","agentId","agentVersionId")})
    identity_path=root/"hosted-identity.json"
    if identity_path.exists() and json.loads(read_secret(identity_path))!=identity:
        raise PermissionError("Hosted volume identity changed; provision a separate service")
    cli.write_private(identity_path,identity)
    cli.write_private(root/"connector.json",config["connection"])
    exports=[]
    for bundle in config["bundles"]:
        language=bundle["language"]; bundle_id=bundle["modelBundleId"]
        request=EvaluationRequest.model_validate({"schemaVersion":"rateloop.evaluator.request.v1",
            "workspaceId":config["workspaceId"],"caseId":"hosted-bootstrap-"+language,
            "idempotencyKey":"hosted-bootstrap-"+language,"modelBundleId":bundle_id,
            "template":overall_approval(language).model_dump(),"input":{"text":"Synthetic registration example.","context":"","evidence":""},"deadlineMs":5000})
        request_path=root/("request-"+language+".json")
        cli.write_private(request_path,request.model_dump())
        try: record=registry.get(bundle_id,config["workspaceId"])
        except KeyError:
            cli.run(Namespace(command="register",state_dir=str(root),model_dir=config["modelDir"],request=str(request_path),
                snapshot_id=None,calibrations=None,real_data=False,selective_policy=None))
        else:
            if record["artifact_root"]!=config["modelDir"] or record["manifest"]["template_commitments"]!=[request.template_commitment()] or record["manifest"]["languages"]!=[language]:
                raise PermissionError("Hosted registration differs from immutable local bundle")
        output=root/"registrations"/(language+".json")
        cli.run(Namespace(command="export-registration",state_dir=str(root),bundle_id=bundle_id,request=str(request_path),output=str(output)))
        exports.append(str(output))
    return {"state":"prepared","registrations":exports,"contentIncluded":False}


class WorkerHealth:
    def __init__(self, clock=time.monotonic):
        self.clock=clock; self.warm=False; self.last_success=None; self.reachable=False; self.stopping=False
        self.lock=threading.Lock()

    def polled(self, healthy: bool):
        with self.lock:
            self.reachable=healthy
            if healthy: self.last_success=self.clock()

    def response(self):
        with self.lock:
            fresh=self.last_success is not None and self.clock()-self.last_success<=150
            healthy=self.warm and fresh and self.reachable and not self.stopping
            return (200 if healthy else 503),{"status":"ready" if healthy else "unavailable"}


@contextmanager
def health_server(health: WorkerHealth, port: int, *, host="0.0.0.0"):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path!="/healthz":
                self.send_error(404); return
            status,body=health.response(); data=json.dumps(body).encode()
            self.send_response(status); self.send_header("Content-Type","application/json")
            self.send_header("Cache-Control","no-store"); self.send_header("Content-Length",str(len(data)))
            self.end_headers(); self.wfile.write(data)
        def log_message(self,*args): pass
    server=ThreadingHTTPServer((host,port),Handler)
    server.daemon_threads=True
    thread=threading.Thread(target=server.serve_forever,name="health",daemon=True); thread.start()
    try: yield server.server_address[1]
    finally:
        health.stopping=True
        server.shutdown(); server.server_close(); thread.join(timeout=2)


def run_hosted(config: dict) -> None:
    offline_environment()
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    bootstrap(config)
    root,local,store,registry=cli.state(Namespace(state_dir=config["stateDir"]))
    c=config["connection"]
    connector=RateLoopConnector(base_url=c["baseUrl"],api_key=c["apiKey"],api_key_id=c["apiKeyId"],
        agent_id=c["agentId"],agent_version_id=c["agentVersionId"],workspace_id=config["workspaceId"],learning=store,
        runtime=RuntimeStore(root/"runtime.sqlite",local["encryptionKey"]),metadata_upload_enabled=True)
    health=WorkerHealth()
    stop=threading.Event()
    previous={s:signal.signal(s,lambda *_:stop.set()) for s in (signal.SIGINT,signal.SIGTERM)}
    try:
        with single_worker(root,config["workerId"]), health_server(health,config["healthPort"]):
            bundles=[b["modelBundleId"] for b in config["bundles"]]
            evaluate=prepare_evaluator(connector,registry,bundles,device="cpu")
            health.warm=True
            worker=OutboundWorker(connector,worker_id=config["workerId"],model_bundle_ids=bundles,evaluate=evaluate,
                poll_seconds=config["pollSeconds"],on_poll=health.polled)
            worker.stop=stop
            worker.run()
    finally:
        health.stopping=True
        connector.close()
        for signum,handler in previous.items(): signal.signal(signum,handler)


def main(argv=None):
    parser=ArgumentParser(description="Private hosted outbound evaluator")
    parser.add_argument("command",choices=("prepare","bootstrap","run"))
    parser.add_argument("--config",required=True)
    args=parser.parse_args(argv)
    try:
        config=read_config(args.config)
        if args.command=="prepare": prepare_model(config)
        elif args.command=="bootstrap": print(json.dumps(bootstrap(config)))
        else: run_hosted(config)
    except Exception as error:
        # Library and configuration exceptions can contain paths, URLs or tokens.
        # A coarse type is enough for operational failure; never print private data.
        print("Hosted evaluator failed ("+type(error).__name__+").",file=__import__("sys").stderr)
        return 1
    return 0


if __name__=="__main__":
    raise SystemExit(main())
