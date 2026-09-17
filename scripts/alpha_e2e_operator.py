"""Privileged synthetic Alpha acceptance driver using the real public CLI/model.

No predictor, label or hosted runtime is replaced. The browser harness supplies
synthetic reviewer responses. Keep its configuration, state and output private.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

from rateloop_evaluator.cli import main as cli, write_private
from rateloop_evaluator.connector import RateLoopConnector
from rateloop_evaluator.learning import LearningStore, read_secret, _digest
from rateloop_evaluator.protocol import EvaluationRequest
from rateloop_evaluator.storage import RuntimeStore
from rateloop_evaluator.worker import OutboundWorker
from rateloop_evaluator.presence import connected_training
from rateloop_evaluator.execution import model_execution


def call(state_dir: Path, *args) -> dict:
    output=io.StringIO()
    with contextlib.redirect_stdout(output):
        code=cli(["--state-dir",str(state_dir),*map(str,args)])
    if code: raise RuntimeError("Operator command failed: "+str(args[0]))
    # Upstream training can print progress before the final CLI result object.
    lines=output.getvalue().splitlines()
    for index in range(len(lines)):
        try:
            result=json.loads("\n".join(lines[index:]))
            if isinstance(result,dict): return result
        except json.JSONDecodeError: pass
    raise RuntimeError("Operator command returned no result")


def connect(config: dict, state_dir: Path) -> RateLoopConnector:
    local=json.loads(read_secret(state_dir/"config.json")); remote=config["connector"]
    return RateLoopConnector(base_url=remote["baseUrl"],api_key=remote["apiKey"],api_key_id=remote["apiKeyId"],
        workspace_id=config["workspaceId"],agent_id=remote["agentId"],agent_version_id=remote["agentVersionId"],
        learning=LearningStore(state_dir/"learning",local["encryptionKey"]),
        runtime=RuntimeStore(state_dir/"runtime.sqlite",local["encryptionKey"]),metadata_upload_enabled=remote["metadataUploadEnabled"],
        allow_insecure_loopback=remote.get("allowInsecureLoopback",False))


def run(args):
    config=json.loads(read_secret(args.config))
    required={"stateDir","workspaceId","modelDir","device","baseBundlePrefix","connector"}
    if set(config)!=required or config["device"] not in ("cpu","mps","cuda"):
        raise ValueError("Operator configuration fields do not match the documented interface")
    state_dir=Path(config["stateDir"]).expanduser().resolve()
    operator_dir=state_dir.parent/(state_dir.name+"-operator")
    connector_file=operator_dir/"connector.json"
    bundles=[config["baseBundlePrefix"]+"-"+language for language in ("en","de")]
    if args.command=="bootstrap":
        call(state_dir,"init","--workspace",config["workspaceId"])
        write_private(connector_file,config["connector"])
        registrations=[]
        for language,bundle_id in zip(("en","de"),bundles):
            request=json.loads((Path(__file__).resolve().parents[1]/f"examples/approval-request-{language}.json").read_text())
            request.update(workspaceId=config["workspaceId"],modelBundleId=bundle_id)
            request_file=operator_dir/(language+"-request.json"); write_private(request_file,request)
            call(state_dir,"register","--model-dir",config["modelDir"],"--request",request_file)
            registration_file=operator_dir/(language+"-registration.json")
            call(state_dir,"export-registration","--bundle-id",bundle_id,"--request",request_file,"--output",registration_file)
            registrations.append(json.loads(read_secret(registration_file)))
        return {"registrations":registrations,"bundles":bundles,"stateDir":str(state_dir),"connectorFile":str(connector_file)}
    if args.command=="worker-once":
        flags=[]
        for bundle_id in args.bundle_id or bundles: flags.extend(["--bundle-id",bundle_id])
        return call(state_dir,"worker","--config",connector_file,"--worker-id","alpha-e2e-mac","--device",config["device"],"--once",*flags)
    connector=connect(config,state_dir)
    try:
        if args.command=="rollback":
            if len(args.bundle_id)!=1: raise ValueError("Choose exactly one rollback bundle")
            with model_execution(connector.learning):
                if connector.sync_grants()["mode"]!="shadow":
                    raise PermissionError("Connected rollback requires current shadow-mode authorization")
                request=EvaluationRequest.model_validate(json.loads(read_secret(operator_dir/(args.language+"-request.json"))))
                expected=args.bundle_id[0]
                result=call(state_dir,"rollback","--template-commitment",request.template_commitment(),
                    "--language",args.language,"--expected-bundle-id",expected)
                if result["bundle_id"]!=expected: raise RuntimeError("Rollback did not restore the expected bundle")
                return {"modelBundleId":expected,"activeModelBundleId":result["bundle_id"],
                    "templateCommitment":result["template_commitment"],"language":result["language"],"mode":result["mode"]}
        connector.sync_grants()
        if args.command=="renewal-check":
            def snapshot():
                with connector.learning.transaction() as database:
                    state=connector._state(database)
                    lease=state.get("authorization_lease")
                    if not lease: raise ValueError("A durable consent execution lease is required")
                    lineage=[{"consentId":identity,"revision":row["consent"]["revision"],"localGrantId":row["local_id"],
                        "authorizationUntil":database["grants"][row["local_id"]]["authorization_until"]}
                        for identity,row in sorted(state.get("consents",{}).items())]
                    return {"leaseId":lease["leaseId"],"issuedAt":lease["issuedAt"],"expiresAt":lease["expiresAt"],"lineage":lineage}
            before=snapshot();time.sleep(.025);connector.sync_grants();after=snapshot()
            return {"before":before,"after":after,"observedFrom":"encrypted_local_permission_mirrors"}
        if args.command=="import-labels":
            return OutboundWorker(connector,worker_id="alpha-e2e-mac",model_bundle_ids=bundles,
                evaluate=lambda _:None).sync_labels()
        if args.command=="verify-erasure":
            present=False
            with connector.learning.transaction() as database:
                deleted=_digest([config["workspaceId"],args.case_id]) in database.get("deleted_cases",{})
                for row in database["evaluations"].values():
                    present=present or (row["workspace_id"]==config["workspaceId"] and row["case_id"]==args.case_id)
                for snapshot in database["snapshots"].values():
                    for part in ("train","calibration","test"):
                        present=present or any(row["workspace_id"]==config["workspaceId"] and row["case_id"]==args.case_id for row in snapshot[part])
                for state in database.get("connectors",{}).values():
                    if state.get("workspace_id")!=config["workspaceId"]: continue
                    for collection in ("results","audits","worker_jobs","receipt_jobs","collections"):
                        present=present or any(row.get("caseId",row.get("case_id"))==args.case_id for row in state.get(collection,{}).values())
            with connector.runtime.connect() as database:
                for table in ("results","outbox","acknowledgments"):
                    for row in database.execute(f"SELECT payload FROM {table}"):
                        payload=connector.runtime.decode(row[0]); result=payload.get("result",payload)
                        present=present or (result.get("workspaceId")==config["workspaceId"] and result.get("caseId")==args.case_id)
            return {"deleted":deleted and not present,"caseId":args.case_id}
        if args.command=="train-candidate":
            # call() invokes cli.main in this thread, so the shared model lock
            # safely spans every stage and remains reentrant in each CLI command.
            with connected_training(connector,worker_id="alpha-e2e-mac",model_bundle_ids=bundles) as check:
                template_request=json.loads(read_secret(operator_dir/(args.language+"-request.json")))
                template_request["modelBundleId"]=args.bundle_id[0]
                request=EvaluationRequest.model_validate(template_request)
                snapshot=call(state_dir,"snapshot","--template",request.template.id,"--version",request.template.version,
                    "--template-commitment",request.template_commitment())
                candidate_dir=operator_dir/("candidate-"+args.bundle_id[0])
                trained=call(state_dir,"train","--snapshot-id",snapshot["snapshotId"],"--model-dir",config["modelDir"],
                    "--output",candidate_dir,"--bundle-id",args.bundle_id[0],"--device",config["device"],"--method","lora","--epochs","1","--max-steps","1")
                check()
                calibrations=operator_dir/(args.bundle_id[0]+"-calibrations.json")
                call(state_dir,"calibrate","--snapshot-id",snapshot["snapshotId"],"--model-dir",trained["modelDir"],
                    "--bundle-id",args.bundle_id[0],"--output",calibrations,"--device",config["device"])
                check()
                request_file=operator_dir/(args.bundle_id[0]+"-request.json"); write_private(request_file,template_request)
                policy=operator_dir/(args.bundle_id[0]+"-policy.json")
                write_private(policy,{"threshold":0.9,"max_false_approval_rate":0.01,"confidence":0.95,"minimum_coverage":0.1})
                call(state_dir,"register","--model-dir",trained["modelDir"],"--request",request_file,
                    "--snapshot-id",snapshot["snapshotId"],"--calibrations",calibrations,"--selective-policy",policy)
                check()
                evidence=operator_dir/(args.bundle_id[0]+"-evidence.json")
                call(state_dir,"score-test","--bundle-id",args.bundle_id[0],"--output",evidence,"--device",config["device"])
                check()
                registration=operator_dir/(args.bundle_id[0]+"-registration.json")
                call(state_dir,"export-registration","--bundle-id",args.bundle_id[0],"--request",request_file,"--output",registration)
                return {"registration":json.loads(read_secret(registration)),"modelBundleId":args.bundle_id[0],
                    "snapshotId":snapshot["snapshotId"],"optimizerSteps":trained["optimizerSteps"],"synthetic":True,
                    "mode":"shadow","qualityClaim":False,"evidenceFile":str(evidence)}
        raise ValueError("Unknown operator command")
    finally: connector.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    commands=parser.add_subparsers(dest="command",required=True)
    commands.add_parser("bootstrap");commands.add_parser("import-labels");commands.add_parser("renewal-check")
    erase=commands.add_parser("verify-erasure");erase.add_argument("--case-id",required=True)
    worker=commands.add_parser("worker-once");worker.add_argument("--bundle-id",action="append")
    train=commands.add_parser("train-candidate");train.add_argument("--language",choices=["en","de"],default="en")
    train.add_argument("--bundle-id",action="append",required=True)
    rollback=commands.add_parser("rollback");rollback.add_argument("--language",choices=["en","de"],default="en")
    rollback.add_argument("--bundle-id",action="append",required=True)
    arguments=parser.parse_args()
    if arguments.command=="train-candidate" and len(arguments.bundle_id)!=1: parser.error("Choose exactly one candidate bundle")
    if arguments.command=="rollback" and len(arguments.bundle_id)!=1: parser.error("Choose exactly one rollback bundle")
    try: print(json.dumps(run(arguments),allow_nan=False))
    except Exception as error:
        print("Alpha operator failed: "+str(error),file=sys.stderr);sys.exit(1)
