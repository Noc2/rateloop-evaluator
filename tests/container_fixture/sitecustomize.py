"""Test-only transport mounted by test_hosted_container.py, never in images."""
import json
import time
from datetime import datetime, timezone
import httpx

OriginalClient=httpx.Client

def handle(request):
    assert request.url.host=="rateloop.example"
    if request.url.path.endswith("/grants"):
        return httpx.Response(200,json={"workspaceId":"container-test","recipientApiKeyId":"test-key",
            "settings":{"mode":"off"},"revocationWatermark":0,"grants":[]})
    if request.url.path.endswith("/workers/heartbeat"):
        body=json.loads(request.content)
        return httpx.Response(200,json={"workerId":body["workerId"],"state":body["state"],
            "lastSeenAt":datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00","Z")})
    raise AssertionError("Paused fixture must never request customer content")

class Client(OriginalClient):
    def __init__(self,*args,**kwargs):
        if str(kwargs.get("base_url","")).startswith("https://rateloop.example"):
            kwargs["transport"]=httpx.MockTransport(handle)
        super().__init__(*args,**kwargs)
httpx.Client=Client
