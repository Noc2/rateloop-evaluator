"""Explicit local lifecycle commands. No training rights are implied by initialization."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import sys
import time

from .learning import LearningStore, provision_key, read_secret
from .protocol import CaseInput, EvaluationRequest, commitment
from .registry import BundleRegistry, provision_signing_key


def read_json(path):
    return json.loads(Path(path).read_text())


def write_private(path, value):
    path = Path(path)
    if path.is_symlink(): raise ValueError("Private output cannot be a symlink")
    path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    fd = os.open(path,os.O_CREAT|os.O_WRONLY|os.O_TRUNC|getattr(os,"O_NOFOLLOW",0),0o600)
    os.fchmod(fd,0o600)
    with os.fdopen(fd,"w") as stream:
        stream.write(json.dumps(value,indent=2,allow_nan=False)+"\n")


def state(args):
    root = Path(args.state_dir).expanduser().resolve()
    config = json.loads(read_secret(root / "config.json"))
    store = LearningStore(root / "learning",config["encryptionKey"])
    return root,config,store,BundleRegistry(store,config["signingKey"])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Private local rating and learning")
    parser.add_argument("--state-dir",default="~/.local/share/rateloop-evaluator")
    commands = parser.add_subparsers(dest="command",required=True)
    init = commands.add_parser("init"); init.add_argument("--workspace",required=True)
    reviewer = commands.add_parser("issue-reviewer-token"); reviewer.add_argument("--reviewer-id",required=True); reviewer.add_argument("--output",required=True)
    provision = commands.add_parser("provision"); provision.add_argument("--model-dir",required=True); provision.add_argument("--revision"); provision.add_argument("--backend",choices=["gliner","gliclass"],default="gliner")
    grant = commands.add_parser("grant"); grant.add_argument("--right",action="append",required=True,choices=["ai_use","private_training","shared_contribution","public_weight_distribution"])
    grant.add_argument("--template",action="append",required=True); grant.add_argument("--hours",type=float,default=24); grant.add_argument("--evidence",required=True)
    revoke = commands.add_parser("revoke"); revoke.add_argument("--grant-id",required=True)
    delete = commands.add_parser("delete-case"); delete.add_argument("--case-id",required=True)
    register = commands.add_parser("register"); register.add_argument("--model-dir",required=True); register.add_argument("--request",required=True)
    register.add_argument("--snapshot-id"); register.add_argument("--calibrations"); register.add_argument("--real-data",action="store_true",help="Declare verified non-synthetic provenance; does not qualify deployment")
    register.add_argument("--selective-policy",help="JSON operating threshold/error policy fixed before final testing")
    export = commands.add_parser("export-registration"); export.add_argument("--bundle-id",required=True); export.add_argument("--request",required=True); export.add_argument("--output",required=True)
    serve = commands.add_parser("serve"); serve.add_argument("--bundle-id",required=True); serve.add_argument("--device",choices=["cpu","mps","cuda"],default="cpu")
    serve.add_argument("--host",default="127.0.0.1"); serve.add_argument("--port",type=int,default=8765); serve.add_argument("--tls-cert"); serve.add_argument("--tls-key")
    snapshot = commands.add_parser("snapshot"); snapshot.add_argument("--template",required=True); snapshot.add_argument("--version",type=int,required=True)
    snapshot.add_argument("--purpose",choices=["private_training","shared_contribution","public_weight_distribution"],default="private_training")
    train = commands.add_parser("train"); train.add_argument("--snapshot-id",required=True); train.add_argument("--model-dir",required=True); train.add_argument("--output",required=True); train.add_argument("--bundle-id",required=True)
    train.add_argument("--device",choices=["cpu","mps","cuda"],default="cpu"); train.add_argument("--method",choices=["full","lora"],default="lora"); train.add_argument("--epochs",type=int,default=3); train.add_argument("--max-steps",type=int,default=-1)
    calibrate = commands.add_parser("calibrate"); calibrate.add_argument("--snapshot-id",required=True); calibrate.add_argument("--model-dir",required=True); calibrate.add_argument("--bundle-id",required=True); calibrate.add_argument("--output",required=True); calibrate.add_argument("--device",choices=["cpu","mps","cuda"],default="cpu")
    score_test = commands.add_parser("score-test"); score_test.add_argument("--bundle-id",required=True); score_test.add_argument("--output",required=True); score_test.add_argument("--device",choices=["cpu","mps","cuda"],default="cpu"); score_test.add_argument("--valid-hours",type=float,default=24)
    promote = commands.add_parser("promote"); promote.add_argument("--bundle-id",required=True); promote.add_argument("--template-commitment",required=True); promote.add_argument("--language",choices=["en","de"],required=True); promote.add_argument("--mode",choices=["shadow","assisted","selective"],default="shadow"); promote.add_argument("--evidence")
    rollback = commands.add_parser("rollback"); rollback.add_argument("--template-commitment",required=True); rollback.add_argument("--language",choices=["en","de"],required=True)
    benchmark = commands.add_parser("benchmark"); benchmark.add_argument("--model-dir",required=True); benchmark.add_argument("--request",required=True); benchmark.add_argument("--device",choices=["cpu","mps","cuda"],default="cpu"); benchmark.add_argument("--iterations",type=int,default=20); benchmark.add_argument("--output",required=True)
    benchmark.add_argument("--backend",choices=["gliner","gliclass"],default="gliner")
    for name in ("worker","install-launchd"):
        command=commands.add_parser(name)
        command.add_argument("--config",required=True,help="Private outbound connector JSON, mode 0600")
        command.add_argument("--worker-id",required=True)
        command.add_argument("--bundle-id",action="append",required=True)
        command.add_argument("--device",choices=["cpu","mps","cuda"],default="cpu")
        command.add_argument("--poll-seconds",type=float,default=5)
        if name=="worker": command.add_argument("--once",action="store_true")
        else:
            command.add_argument("--output",default="~/Library/LaunchAgents/ai.rateloop.evaluator.worker.plist")
            command.add_argument("--load",action="store_true",help="Load the installed worker into this macOS login session")
    for name in ("connect-sync","connect-flush","connect-evaluate","connect-import-labels","connect-release"):
        command = commands.add_parser(name); command.add_argument("--config",required=True,help="Private connector JSON, mode 0600")
        if name == "connect-evaluate":
            command.add_argument("--request",required=True); command.add_argument("--review-context",required=True)
            command.add_argument("--endpoint",default="http://127.0.0.1:8765"); command.add_argument("--client-credentials")
            command.add_argument("--allow-offline",action="store_true",help="Permit local evaluation without claiming a blind audit when RateLoop is unreachable")
        if name == "connect-import-labels":
            command.add_argument("--grant-id",required=True); command.add_argument("--question-id",required=True)
            command.add_argument("--template-commitment",required=True); command.add_argument("--positive-label",required=True); command.add_argument("--negative-label",required=True)
        if name == "connect-release": command.add_argument("--input-commitment",required=True)
    args = parser.parse_args(argv)
    try:
        result = run(args)
        if result is not None: print(json.dumps(result,indent=2,allow_nan=False))
    except (ValueError,PermissionError,KeyError,FileNotFoundError,RuntimeError) as error:
        # Local CLI errors contain no case text; upstream training diagnostics stay local.
        print(f"Evaluator: {error}",file=sys.stderr)
        return 1
    return 0


def run(args):
    if args.command == "init":
        root = Path(args.state_dir).expanduser().resolve()
        if root.exists() and any(root.iterdir()): raise ValueError("Initialization requires an empty directory")
        root.mkdir(parents=True,exist_ok=True,mode=0o700); os.chmod(root,0o700)
        provision_key(root / "encryption.key"); provision_signing_key(root / "signing.key")
        token = secrets.token_urlsafe(32)
        write_private(root / "client.json",{"token":token,"workspaceId":args.workspace})
        write_private(root / "config.json",{"workspaceId":args.workspace,"encryptionKey":str(root / "encryption.key"),
            "signingKey":str(root / "signing.key"),"tokens":[{"sha256":hashlib.sha256(token.encode()).hexdigest(),"workspaceId":args.workspace,"roles":["evaluate"]}]})
        return {"stateDirectory":str(root),"clientCredentials":str(root / "client.json"),"grants":[],"next":"Explicitly grant AI use before evaluation; training is separately authorized."}
    if args.command == "provision":
        from .backends import MODEL_REVISION, GLICLASS_MODEL_REVISION, provision_model, provision_gliclass_model
        if args.backend == "gliclass":
            return provision_gliclass_model(args.model_dir,revision=args.revision or GLICLASS_MODEL_REVISION)
        return provision_model(args.model_dir,revision=args.revision or MODEL_REVISION)
    if args.command == "benchmark":
        from .backends import GLiNERBackend, GLiClassBackend
        from .benchmark import benchmark_backend
        req = EvaluationRequest.model_validate(read_json(args.request))
        backend_type = GLiClassBackend if args.backend == "gliclass" else GLiNERBackend
        report = benchmark_backend(backend_type(args.model_dir,args.device),[{"text":req.input.render(),"questions":[q.model_dump() for q in req.template.questions]}],iterations=args.iterations)
        write_private(args.output,report)
        return {"report":str(Path(args.output).resolve()),"qualityClaim":False}
    root,config,store,registry = state(args); workspace = config["workspaceId"]
    if args.command == "install-launchd":
        from .worker import install_launchd
        return install_launchd(state_dir=root,config_path=args.config,worker_id=args.worker_id,bundle_ids=args.bundle_id,
            device=args.device,poll_seconds=args.poll_seconds,output=args.output,load=args.load)
    if args.command.startswith("connect-") or args.command == "worker":
        import httpx
        from urllib.parse import urlsplit
        from .connector import RateLoopConnector, ConnectorUnavailable
        from .storage import RuntimeStore
        connection = json.loads(read_secret(args.config))
        required = {"baseUrl","apiKey","apiKeyId","agentId","agentVersionId","metadataUploadEnabled"}
        if not isinstance(connection,dict) or set(connection)-required-{"allowInsecureLoopback"} or not required <= set(connection):
            raise ValueError("Connector configuration fields do not match the documented interface")
        connector = RateLoopConnector(base_url=connection["baseUrl"],api_key=connection["apiKey"],api_key_id=connection["apiKeyId"],
            agent_id=connection["agentId"],agent_version_id=connection["agentVersionId"],workspace_id=workspace,learning=store,
            runtime=RuntimeStore(root / "runtime.sqlite",config["encryptionKey"]),metadata_upload_enabled=connection["metadataUploadEnabled"],
            allow_insecure_loopback=connection.get("allowInsecureLoopback",False))
        try:
            if args.command == "worker":
                from .backends import GLiNERBackend
                from .service import Principal, create_app
                from .worker import OutboundWorker, run_worker
                apps={}
                identity=Principal(workspace,frozenset({"evaluate"}))
                def evaluate_job(request):
                    bundle_id=request.modelBundleId
                    if bundle_id not in apps:
                        record=registry.get(bundle_id,workspace)
                        backend=GLiNERBackend(record["artifact_root"],args.device)
                        backend.load()
                        def validate(value):
                            registry.get(bundle_id,workspace,verify_artifacts=False)
                            active=registry.active(workspace,value.template_commitment(),value.template.language,verify_artifacts=False)
                            if active["bundle_id"] != bundle_id: raise PermissionError("Queued model is no longer active")
                            return active
                        apps[bundle_id]=create_app(backend=backend,bundle=record["manifest"],learning=store,runtime=connector.runtime,
                            tokens={"0"*64:identity},validate_bundle=validate)
                    return apps[bundle_id].state.evaluate(request,identity)
                return run_worker(OutboundWorker(connector,worker_id=args.worker_id,model_bundle_ids=args.bundle_id,
                    evaluate=evaluate_job,poll_seconds=args.poll_seconds),root,once=args.once)
            if args.command == "connect-sync": return connector.sync_grants()
            if args.command == "connect-flush": return connector.flush()
            if args.command == "connect-release": return connector.release_result(args.input_commitment)
            if args.command == "connect-import-labels":
                return connector.fetch_and_import_labels(args.grant_id,question_id=args.question_id,template_commitment=args.template_commitment,
                    outcome_labels={"positive":args.positive_label,"negative":args.negative_label})
            try:
                status = connector.sync_grants()
                if status["mode"] != "shadow": raise PermissionError("Connected evaluator is off or paused")
            except ConnectorUnavailable:
                if not args.allow_offline: raise
            endpoint = urlsplit(args.endpoint)
            if endpoint.username or endpoint.password or endpoint.query or endpoint.fragment or endpoint.path not in ("","/"):
                raise ValueError("Local evaluator endpoint must be an origin")
            if endpoint.scheme != "https" and not (endpoint.scheme == "http" and endpoint.hostname in ("localhost","127.0.0.1","::1")):
                raise ValueError("Remote evaluator connections require HTTPS")
            credentials = json.loads(read_secret(args.client_credentials or root / "client.json"))
            if credentials["workspaceId"] != workspace: raise PermissionError("Local client credential belongs to another workspace")
            request = EvaluationRequest.model_validate(read_json(args.request))
            with httpx.Client(base_url=args.endpoint,headers={"Authorization":"Bearer "+credentials["token"]},trust_env=False,follow_redirects=False,timeout=65) as client:
                def evaluate(value):
                    with client.stream("POST","/v1/evaluate",json=value.model_dump()) as response:
                        if response.status_code != 200: raise RuntimeError(f"Local evaluator returned {response.status_code}; human review remains required")
                        chunks = []; size = 0
                        for chunk in response.iter_bytes():
                            size += len(chunk)
                            if size > 100_000: raise ValueError("Local evaluator response exceeds limit")
                            chunks.append(chunk)
                        return json.loads(b"".join(chunks))
                return connector.run_with_audit(request,evaluate,read_json(args.review_context),allow_offline=args.allow_offline)
        finally: connector.close()
    if args.command == "issue-reviewer-token":
        token = secrets.token_urlsafe(32)
        config["tokens"].append({"sha256":hashlib.sha256(token.encode()).hexdigest(),"workspaceId":workspace,"roles":["feedback"],"annotatorId":args.reviewer_id})
        write_private(root / "config.json",config)
        write_private(args.output,{"token":token,"workspaceId":workspace,"annotatorId":args.reviewer_id})
        return {"reviewerCredentials":str(Path(args.output).resolve()),"restartRequired":True,"purpose":"Trusted human feedback only; keep separate from inference agents."}
    if args.command == "grant":
        if not 0 < args.hours <= 24*30: raise ValueError("Local grants must expire within 30 days")
        return store.add_grant(workspace_id=workspace,rights=args.right,expires_at=time.time()+args.hours*3600,
            template_ids=args.template,fields=["input.text","input.context","input.evidence","human_labels"],evidence=args.evidence)
    if args.command == "revoke": return store.revoke_grant(args.grant_id,workspace)
    if args.command == "delete-case":
        from .storage import RuntimeStore
        deleted = store.delete_case(workspace,args.case_id)
        deleted["runtimeDeleted"] = RuntimeStore(root / "runtime.sqlite",config["encryptionKey"]).delete_case(workspace,args.case_id)
        return deleted
    if args.command == "snapshot":
        snapshot = store.create_snapshot(workspace,args.template,args.version,purpose=args.purpose)
        return {"snapshotId":snapshot["id"],"groups":snapshot["group_count"],"purpose":snapshot["purpose"]}
    if args.command == "train":
        from .training import TrainOptions, train_snapshot
        report = train_snapshot(store,args.snapshot_id,workspace,args.model_dir,args.output,bundle_id=args.bundle_id,
            options=TrainOptions(method=args.method,device=args.device,epochs=args.epochs,max_steps=args.max_steps))
        return {"modelDir":report["modelDir"],"optimizerSteps":report["training"]["optimizerSteps"],"qualityClaim":False}
    if args.command == "calibrate":
        from .backends import GLiNERBackend, validate_local_model
        from .calibration import fit_temperature
        snapshot = store.load_snapshot(args.snapshot_id,workspace)
        model_files = validate_local_model(args.model_dir)
        if model_files.get("training",{}).get("bundleId") != args.bundle_id or model_files.get("training",{}).get("snapshotId") != args.snapshot_id or model_files.get("training",{}).get("workspaceId") != workspace:
            raise ValueError("Calibration requires the matching trained bundle and snapshot")
        backend = GLiNERBackend(args.model_dir,args.device)
        representatives = {}
        for row in sorted(snapshot["calibration"],key=lambda r:r["evaluation_id"]): representatives.setdefault(row["group_id"],row)
        rows = list(representatives.values())
        if not rows: raise ValueError("Calibration groups are required")
        predictions = []
        for row in rows:
            text = CaseInput.model_validate(row["input"]).render()
            if backend.count_tokens(text,row["template"]["questions"]) > row["template"]["maxTokens"]:
                raise ValueError("Calibration input exceeds its template token limit")
            predictions.append(backend.predict(text,row["template"]["questions"]))
        artifacts = [fit_temperature([p[q["id"]] for p in predictions],[r["labels"][q["id"]] for r in rows],model_bundle_id=args.bundle_id,
            template_commitment=rows[0]["template_commitment"],question_id=q["id"],language=rows[0]["template"]["language"],example_ids=[r["group_id"] for r in rows],model_weights_sha256=model_files["files"]["model.safetensors"]) for q in rows[0]["template"]["questions"]]
        store.load_snapshot(args.snapshot_id,workspace)
        write_private(args.output,artifacts)
        return {"calibrations":str(Path(args.output).resolve()),"groups":len(rows),"qualityClaim":False}
    if args.command == "score-test":
        from .backends import GLiNERBackend
        if not 0 < args.valid_hours <= 30*24: raise ValueError("Evidence validity must be within 30 days")
        record = registry.get(args.bundle_id,workspace); manifest = record["manifest"]
        if not manifest.get("snapshot_id") or not manifest.get("selective_policy"):
            raise ValueError("Register the trained bundle with a predeclared selective policy first")
        snapshot = store.load_snapshot(manifest["snapshot_id"],workspace)
        representatives = {}
        for row in sorted(snapshot["test"],key=lambda r:r["evaluation_id"]): representatives.setdefault(row["group_id"],row)
        if not representatives: raise ValueError("Independent test groups are required")
        backend = GLiNERBackend(record["artifact_root"],args.device)
        rows = []; first = next(iter(representatives.values()))
        for row in representatives.values():
            if row["template_commitment"] != first["template_commitment"]: raise ValueError("Test scope must use one immutable template")
            text = CaseInput.model_validate(row["input"]).render(); questions = row["template"]["questions"]
            if backend.count_tokens(text,questions) > manifest["max_tokens"]: raise ValueError("Test evidence exceeds token limit")
            rows.append({"evaluation_id":row["evaluation_id"],"raw_scores":backend.predict(text,questions)})
        registry.get(args.bundle_id,workspace); store.load_snapshot(manifest["snapshot_id"],workspace)
        now = time.time()
        evidence = {"template_commitment":first["template_commitment"],"language":first["template"]["language"],"synthetic":manifest["synthetic"],
            "observed_at":now,"valid_until":now+args.valid_hours*3600,**manifest["selective_policy"],"rows":rows}
        write_private(args.output,evidence)
        return {"evidenceFile":str(Path(args.output).resolve()),"groups":len(rows),"qualityClaim":False,"next":"Promotion recomputes the acceptance bound; scoring alone does not enable automation."}
    if args.command == "register":
        from .backends import MANIFEST_NAME, file_hash, validate_local_model
        req = EvaluationRequest.model_validate(read_json(args.request))
        if req.workspaceId != workspace: raise ValueError("Request workspace does not match local configuration")
        source = validate_local_model(args.model_dir)
        training = source.get("training")
        if training:
            if training.get("bundleId") != req.modelBundleId or training.get("workspaceId") != workspace:
                raise ValueError("Trained model identity does not match this workspace and bundle")
            if not args.snapshot_id or training.get("snapshotId") != args.snapshot_id:
                raise ValueError("Trained model requires its matching --snapshot-id lineage")
        manifest = {"id":req.modelBundleId,"model_id":source["source"]["repository"],"model_revision":source["source"]["revision"],
            "files":{**source["files"],MANIFEST_NAME:file_hash(Path(args.model_dir) / MANIFEST_NAME)},"template_commitments":[req.template_commitment()],"languages":[req.template.language],
            "calibrations":read_json(args.calibrations) if args.calibrations else [],"synthetic":not args.real_data,"max_tokens":req.template.maxTokens}
        if args.snapshot_id: manifest["snapshot_id"] = args.snapshot_id
        if args.selective_policy: manifest["selective_policy"] = read_json(args.selective_policy)
        registry.register(manifest,workspace,args.model_dir)
        registry.promote(req.modelBundleId,workspace,template_commitment=req.template_commitment(),language=req.template.language,mode="shadow")
        return {"modelBundleId":req.modelBundleId,"templateCommitment":req.template_commitment(),"mode":"shadow","publicKey":registry.public_key}
    if args.command == "export-registration":
        from .backends import MANIFEST_NAME
        record = registry.get(args.bundle_id,workspace); manifest = record["manifest"]
        req = EvaluationRequest.model_validate(read_json(args.request))
        if req.workspaceId != workspace or req.modelBundleId != args.bundle_id or req.template_commitment() not in manifest["template_commitments"]:
            raise ValueError("Registration request does not match the signed bundle")
        model = read_json(Path(record["artifact_root"]) / MANIFEST_NAME)
        if model.get("training") and not model["source"].get("baseWeightsSha256"):
            raise ValueError("Trained model is missing its original weight digest")
        base_hash = model["source"].get("baseWeightsSha256",model["files"].get("model.safetensors"))
        if not base_hash: raise ValueError("Original model weight digest is unavailable")
        active = registry.active(workspace,req.template_commitment(),req.template.language)
        if active["bundle_id"] != args.bundle_id: raise ValueError("Only the active bundle can be exported for registration")
        calibrations = {c["question_id"]:c for c in manifest["calibrations"] if c["template_commitment"] == req.template_commitment() and c["language"] == req.template.language}
        # No calibration is asserted to SaaS until a time-bounded deployment gate exists.
        expiry = (active.get("gate") or {}).get("valid_until")
        if expiry is not None:
            from datetime import datetime, timezone
            expiry = datetime.fromtimestamp(expiry,timezone.utc).isoformat(timespec="milliseconds").replace("+00:00","Z")
        criteria = []
        for question in req.template.questions:
            cal = calibrations.get(question.id) if expiry else None
            criteria.append({"questionId":question.id,"labels":[label.id for label in question.labels],"calibrationId":cal["id"] if cal else None,
                "calibrationCommitment":commitment(cal,"rateloop.calibration.v1") if cal else None,"calibrationExpiresAt":expiry if cal else None})
        snapshot_digest = None
        if manifest.get("snapshot_id"): snapshot_digest = "sha256:"+store.load_snapshot(manifest["snapshot_id"],workspace)["content_digest"]
        registration = {"modelBundleId":args.bundle_id,"templateCommitment":req.template_commitment(),"language":req.template.language,
            "baseWeightsCommitment":"sha256:"+base_hash,"adapterCommitment":commitment(model["files"],"rateloop.adaptation.v1") if model.get("training") else None,
            "tokenizerCommitment":commitment({k:v for k,v in model["files"].items() if "tokenizer" in k},"rateloop.tokenizer.v1"),"quantization":"fp32",
            "trainingSnapshotCommitment":snapshot_digest,"evaluationReportCommitment":commitment(active,"rateloop.deployment-evidence.v1"),
            "licenseManifestCommitment":commitment({"software":"Apache-2.0","weights":model["source"].get("license","Apache-2.0"),"model":manifest["model_id"],"revision":manifest["model_revision"]},"rateloop.licenses.v1"),
            "maxTokens":manifest["max_tokens"],"criteria":criteria}
        write_private(args.output,registration)
        return {"registrationFile":str(Path(args.output).resolve()),"contentIncluded":False,"mode":active["mode"]}
    if args.command == "promote":
        return registry.promote(args.bundle_id,workspace,template_commitment=args.template_commitment,language=args.language,mode=args.mode,evidence=read_json(args.evidence) if args.evidence else None)
    if args.command == "rollback": return registry.rollback(workspace,args.template_commitment,args.language)
    if args.command == "serve":
        import uvicorn
        from .backends import GLiNERBackend
        from .service import Principal, create_app
        from .storage import RuntimeStore
        if args.host not in ("127.0.0.1","::1","localhost") and not (args.tls_cert and args.tls_key):
            raise ValueError("Non-loopback serving requires TLS certificate and key")
        record = registry.get(args.bundle_id,workspace)
        backend = GLiNERBackend(record["artifact_root"],args.device)
        backend.load()
        def validate(request):
            registry.get(args.bundle_id,workspace,verify_artifacts=False)
            active = registry.active(workspace,request.template_commitment(),request.template.language,verify_artifacts=False)
            if active["bundle_id"] != args.bundle_id: raise PermissionError("Active model changed; restart the worker")
            return active
        app = create_app(backend=backend,bundle=record["manifest"],learning=store,runtime=RuntimeStore(root / "runtime.sqlite",config["encryptionKey"]),
            tokens={token["sha256"]:Principal(token["workspaceId"],frozenset(token["roles"]),token.get("annotatorId")) for token in config["tokens"]},validate_bundle=validate)
        uvicorn.run(app,host=args.host,port=args.port,ssl_certfile=args.tls_cert,ssl_keyfile=args.tls_key,access_log=False,workers=1)
        return None
    raise ValueError("Unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
