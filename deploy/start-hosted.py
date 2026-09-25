"""Volume ownership only as root; provisioning and inference run unprivileged."""
import json
import os
from pathlib import Path
import subprocess
import sys


def start():
    os.umask(0o077)
    volume=Path("/data")
    if volume.is_symlink(): raise ValueError("Invalid volume")
    volume.mkdir(mode=0o700,exist_ok=True)
    if os.getuid()==0:
        os.chown(volume,10001,10001); os.chmod(volume,0o700)
        os.setgroups([]); os.setgid(10001); os.setuid(10001)
    if os.getuid()!=10001: raise PermissionError("Unexpected runtime identity")
    os.environ["HOME"]="/home/evaluator"
    from rateloop_evaluator.cli import write_private
    from rateloop_evaluator.hosted import read_config, emit_registrations
    path=volume/"hosted.json"
    encoded=os.environ.pop("RATELOOP_HOSTED_CONFIG_JSON",None)
    if encoded is not None:
        if len(encoded)>20000: raise ValueError("Configuration too large")
        write_private(path,json.loads(encoded))
    config=read_config(path)
    if (config["modelDir"],config["stateDir"])!=("/data/models/gliner25","/data/state"):
        raise ValueError("Hosted container paths must use its durable volume")
    if config["healthPort"]!=int(os.environ.get("PORT","8080")):
        raise ValueError("Health port does not match service port")
    provision=os.environ.pop("RATELOOP_PROVISION_MODEL","0")
    if provision not in ("0","1"): raise ValueError("Invalid provisioning selection")
    export=os.environ.pop("RATELOOP_HOSTED_EXPORT_REGISTRATIONS","0")
    if export not in ("0","1"): raise ValueError("Invalid registration export selection")
    if provision=="1":
        env=dict(os.environ)
        # Explicit provisioning is a separate process; offline inference never
        # imports a model downloader or accesses an upstream model repository.
        env.pop("HF_HUB_OFFLINE",None); env.pop("TRANSFORMERS_OFFLINE",None)
        subprocess.run([sys.executable,"-m","rateloop_evaluator.hosted","prepare","--config",str(path)],env=env,check=True)
    if export=="1": emit_registrations(config)
    os.execv(sys.executable,[sys.executable,"-m","rateloop_evaluator.hosted","run","--config",str(path)])


if __name__=="__main__":
    try: start()
    except Exception as error:
        print("Hosted startup failed ("+type(error).__name__+").",file=sys.stderr)
        raise SystemExit(1)
