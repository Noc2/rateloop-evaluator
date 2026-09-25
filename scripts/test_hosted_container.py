"""Opt-in real Linux model/startup/restart check against a test-only transport.

Usage: python scripts/test_hosted_container.py IMAGE /absolute/public/model/path
Requires Docker; never sends a request to RateLoop or a model provider.
"""
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

image,model=sys.argv[1:]
root=Path(__file__).resolve().parents[1]
name="rateloop-hosted-test-"+uuid.uuid4().hex[:10]
volume=name+"-state"

def docker(*args,check=True):
    return subprocess.run(["docker",*args],check=check,capture_output=True,text=True)

config={"schemaVersion":"rateloop.hosted-worker.v1","workspaceId":"container-test","workerId":"hosted-test",
    "modelDir":"/data/models/gliner25","stateDir":"/data/state",
    "bundles":[{"language":"en","modelBundleId":"hosted-test-en"},{"language":"de","modelBundleId":"hosted-test-de"}],
    "connection":{"baseUrl":"https://rateloop.example","apiKey":"dummy-test-key-never-authorized","apiKeyId":"test-key",
        "agentId":"test-agent","agentVersionId":"test-version","metadataUploadEnabled":True},"pollSeconds":1,"healthPort":8080}
probe='''import json,os,urllib.request,hashlib,pathlib
root=pathlib.Path("/data/state")
try:
 response=urllib.request.urlopen("http://127.0.0.1:8080/healthz",timeout=2)
 status=response.status
except Exception: status=503
p=pathlib.Path("/proc/1/status").read_text()
print(json.dumps({"status":status,"uid":[s for s in p.splitlines() if s.startswith("Uid:")][0],"key":hashlib.sha256((root/"encryption.key").read_bytes()).hexdigest() if (root/"encryption.key").exists() else None,"memoryCurrent":pathlib.Path("/sys/fs/cgroup/memory.current").read_text().strip(),"memoryEvents":pathlib.Path("/sys/fs/cgroup/memory.events").read_text().strip()}))
'''
def wait_ready():
    started=time.monotonic(); saw_unready=False
    for _ in range(90):
        state=json.loads(docker("inspect",name,"--format","{{json .State}}").stdout)
        if not state["Running"]:
            raise RuntimeError("Container stopped: "+docker("logs",name).stdout)
        value=json.loads(docker("exec",name,"python","-c",probe).stdout)
        if value["status"]==200:
            assert "10001" in value["uid"]
            assert "oom_kill 0" in value["memoryEvents"]
            assert saw_unready,"Readiness was not observed before warmup"
            print(json.dumps({"readyAfterSeconds":round(time.monotonic()-started,2),**value}),flush=True)
            return value["key"]
        saw_unready=True; time.sleep(1)
    raise RuntimeError("Warmup failed: "+docker("logs",name).stderr)

try:
    docker("volume","create",volume)
    docker("run","-d","--name",name,"--platform","linux/amd64","--network","none","--cpus","1",
        "--memory","2500000000","--memory-swap","2500000000","-v",volume+":/data",
        "-v",str(Path(model).resolve())+":/data/models/gliner25:ro",
        "-v",str(root/"tests/container_fixture")+":/testing:ro",
        "-e","PYTHONPATH=/testing","-e","RATELOOP_HOSTED_EXPORT_REGISTRATIONS=1",
        "-e","RATELOOP_HOSTED_CONFIG_JSON="+json.dumps(config),image)
    first=wait_ready()
    logs=docker("logs",name).stdout
    records=[json.loads(line.split(" ",1)[1]) for line in logs.splitlines() if line.startswith("RATELOOP_HOSTED_REGISTRATION_V1 ")]
    assert len(records)==2
    for record in records:
        stored=json.loads(docker("exec",name,"cat","/data/state/registrations/"+record["language"]+".json").stdout)
        assert record==stored
    assert config["connection"]["apiKey"] not in logs and config["workspaceId"] not in logs
    started=time.monotonic();docker("stop","--time","20",name)
    state=json.loads(docker("inspect",name,"--format","{{json .State}}").stdout)
    assert state["ExitCode"]==0 and not state["OOMKilled"]
    print(json.dumps({"stoppedSeconds":round(time.monotonic()-started,2),"exitCode":state["ExitCode"]}),flush=True)
    docker("start",name)
    assert wait_ready()==first,"Restart rotated the encryption identity"
    restarted=[json.loads(line.split(" ",1)[1]) for line in docker("logs",name).stdout.splitlines() if line.startswith("RATELOOP_HOSTED_REGISTRATION_V1 ")]
    assert restarted==records+records,"Restart changed registration evidence"
    docker("stop","--time","20",name)
    print(json.dumps({"result":"passed","model":"real offline pinned GLiNER","remote":"synthetic paused transport","budget":"1 CPU / 2500000000 bytes / no swap"}),flush=True)
finally:
    docker("rm","-f",name,check=False)
    docker("volume","rm",volume,check=False)
