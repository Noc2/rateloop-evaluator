"""One installed connector for independently enabled ratings and generation."""
from __future__ import annotations

import json
from pathlib import Path
import signal
import threading

from . import cli
from .connector import RateLoopConnector, ConnectorUnavailable, ConnectorRejected
from .execution import model_execution, ExecutionBusy
from .generation_worker import GenerationWorker
from .learning import read_secret
from .ollama import OllamaRuntime, OllamaError
from .storage import RuntimeStore
from .worker import OutboundWorker, install_launchd, single_worker
from .worker_runtime import prepare_evaluator


def install_paired(args):
    return install_launchd(state_dir=Path(args.state_dir), config_path=args.config, worker_id=args.worker_id,
        bundle_ids=args.bundle_id, device=args.device, poll_seconds=args.poll_seconds, output=args.output, load=args.load, paired=True)


def run_paired(args):
    root,local,store,registry=cli.state(args)
    connection=json.loads(read_secret(args.config))
    g=args.generation
    if not isinstance(g,dict) or set(g)!={"baseUrl","model"} or not isinstance(g['model'],dict):
        raise ValueError("Invalid local generation configuration")
    runtime=OllamaRuntime(model=g['model']['model'],base_url=g['baseUrl'],context_tokens=g['model']['contextTokens'],expected_identity=g['model'])
    generation=GenerationWorker(connection=connection,worker_id=args.worker_id,model=g['model'],runtime=runtime,
        outbox=RuntimeStore(root/'generation.sqlite',local['encryptionKey']))
    connector=None;rating=None;evaluate=None;threads=[]
    try:
        if args.bundle_id:
            connector=RateLoopConnector(base_url=connection['baseUrl'],api_key=connection['apiKey'],api_key_id=connection['apiKeyId'],
                agent_id=connection['agentId'],agent_version_id=connection['agentVersionId'],workspace_id=local['workspaceId'],
                learning=store,runtime=RuntimeStore(root/'runtime.sqlite',local['encryptionKey']),metadata_upload_enabled=True,
                allow_insecure_loopback=connection.get('allowInsecureLoopback',False))
            evaluate=prepare_evaluator(connector,registry,args.bundle_id,device=args.device)
            rating=OutboundWorker(connector,worker_id=args.worker_id,model_bundle_ids=args.bundle_id,evaluate=evaluate,poll_seconds=args.poll_seconds)
        with single_worker(root,args.worker_id):
            if args.once:
                results={}
                if rating: results['evaluation']=rating.run_once()
                with model_execution(store): results['generation']=generation.run_once()
                return results
            def stop(*_):
                generation.stop.set()
                if rating: rating.stop.set()
            previous={s:signal.signal(s,stop) for s in (signal.SIGINT,signal.SIGTERM)}
            if rating:
                thread=threading.Thread(target=rating.run,name='rating-worker',daemon=True);thread.start();threads.append(thread)
            try:
                while not generation.stop.is_set():
                    try:
                        with model_execution(store): generation.run_once()
                    except (ConnectorUnavailable,ConnectorRejected,PermissionError,ValueError,OllamaError,ExecutionBusy): pass
                    generation.stop.wait(args.poll_seconds)
            finally:
                stop()
                for thread in threads: thread.join(timeout=70)
                for s,handler in previous.items(): signal.signal(s,handler)
        return {'state':'stopped'}
    finally:
        generation.close()
        if evaluate: evaluate.close()
        if connector: connector.close()
