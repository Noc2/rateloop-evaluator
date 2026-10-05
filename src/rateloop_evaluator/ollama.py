"""Bounded, explicitly pinned local Ollama inference; no provisioning or fallback."""
from __future__ import annotations

import ipaddress
import json
import re
import threading
import time
from urllib.parse import urlsplit

import httpx

from .protocol import commitment

_MODEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}$")
_DIGEST = re.compile(r"^sha256:[a-f0-9]{64}$")


class OllamaError(RuntimeError):
    def __init__(self, code="generation_failed"):
        if code not in {"model_unavailable", "context_overflow", "generation_failed", "output_limit"}:
            code = "generation_failed"
        self.code = code
        super().__init__(code)


def _bounded_object(response, limit=1_000_000):
    data=bytearray()
    for chunk in response.iter_bytes(chunk_size=4096):
        if len(data)+len(chunk)>limit: raise OllamaError()
        data.extend(chunk)
    try: value=json.loads(data)
    except (ValueError, UnicodeError): raise OllamaError() from None
    if not isinstance(value,dict): raise OllamaError()
    return value


class OllamaRuntime:
    """The operator chooses the endpoint/model; jobs cannot change either.

    templateDigest commits {manifestDigest, template, system, parameters, messages}
    with RFC8785 under rateloop.ollama.template.v1. weightDigest is the model's
    local FROM blob SHA-256, not the differently scoped model-manifest digest.
    """
    def __init__(self, *, model: str, base_url="http://127.0.0.1:11434", context_tokens=8192,
                 expected_identity: dict | None = None, transport=None):
        url=urlsplit(base_url)
        try: loopback=ipaddress.ip_address(url.hostname or "").is_loopback
        except ValueError: loopback=url.hostname=="localhost"
        if (not loopback or url.scheme not in ("http","https") or url.username or url.password
                or url.path not in ("","/") or url.query or url.fragment):
            raise ValueError("Ollama endpoint must be an explicitly selected loopback origin")
        if not isinstance(model,str) or not _MODEL.fullmatch(model) or "cloud" in model.lower().split(":")[-1]:
            raise ValueError("Select a provisioned local Ollama model")
        if type(context_tokens) is not int or not 2048<=context_tokens<=32768:
            raise ValueError("Local context must be 2048-32768 tokens")
        self.model=model;self.context_tokens=context_tokens;self.expected_identity=expected_identity
        self.client=httpx.Client(base_url=base_url.rstrip("/"),timeout=httpx.Timeout(10,read=5),trust_env=False,
                                 follow_redirects=False,transport=transport)

    def close(self): self.client.close()

    def _request(self, method, path, **kwargs):
        try:
            with self.client.stream(method,path,**kwargs) as response:
                if response.status_code!=200: raise OllamaError("model_unavailable")
                return _bounded_object(response)
        except httpx.HTTPError: raise OllamaError("model_unavailable") from None

    def identity(self) -> dict:
        version=self._request("GET","/api/version").get("version")
        # Earlier API versions may silently ignore no-truncation/render options.
        if not isinstance(version,str) or not re.fullmatch(r"0\.35\.[0-9]+",version):
            raise OllamaError("model_unavailable")
        models=self._request("GET","/api/tags").get("models")
        if not isinstance(models,list): raise OllamaError("model_unavailable")
        matches=[m for m in models if isinstance(m,dict) and m.get("name")==self.model]
        if len(matches)!=1: raise OllamaError("model_unavailable")
        item=matches[0]
        manifest=item.get("digest","")
        if isinstance(manifest,str) and re.fullmatch(r"[a-f0-9]{64}",manifest): manifest="sha256:"+manifest
        if not isinstance(manifest,str) or not _DIGEST.fullmatch(manifest): raise OllamaError("model_unavailable")
        show=self._request("POST","/api/show",json={"model":self.model})
        if any(value.get("remote_host") or value.get("remote_model") for value in (item,show)):
            raise OllamaError("model_unavailable")
        capabilities=show.get("capabilities",[])
        if not isinstance(capabilities,list) or "completion" not in capabilities: raise OllamaError("model_unavailable")
        modelfile=show.get("modelfile","")
        if not isinstance(modelfile,str): raise OllamaError("model_unavailable")
        source=re.search(r"^FROM\s+.*[/\\]sha256[-:]([a-f0-9]{64})\s*$",modelfile,re.MULTILINE)
        if source is None: raise OllamaError("model_unavailable")
        info=show.get("model_info") or {}
        if not isinstance(info,dict): raise OllamaError("model_unavailable")
        contexts=[v for k,v in info.items() if k.endswith(".context_length") and type(v) is int]
        if not contexts or self.context_tokens>max(contexts): raise OllamaError("context_overflow")
        quantization=(show.get("details") or {}).get("quantization_level")
        if not isinstance(quantization,str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,40}",quantization):
            raise OllamaError("model_unavailable")
        template={"manifestDigest":manifest,**{k:show.get(k,"" if k!="messages" else []) for k in ("template","system","parameters","messages")}}
        if not isinstance(template["template"],str) or not template["template"]:
            raise OllamaError("model_unavailable")
        result={"schemaVersion":"rateloop.generation-model.v1","model":self.model,"weightDigest":"sha256:"+source[1],
                "templateDigest":commitment(template,"rateloop.ollama.template.v1"),"runtime":"ollama","runtimeVersion":version,
                "quantization":quantization,"contextTokens":self.context_tokens}
        if self.expected_identity is not None and result!=self.expected_identity:
            raise OllamaError("model_unavailable")
        return result

    def generate(self, messages: list[dict], *, max_output_tokens=2048, max_output_characters=16000,
                 timeout_seconds=180, on_progress=None, cancelled: threading.Event | None = None,
                 output_schema: dict | None = None) -> dict:
        if (not isinstance(messages,list) or not 1<=len(messages)<=64 or any(not isinstance(m,dict)
                or set(m)!={"role","content"} or m["role"] not in ("system","user","assistant")
                or not isinstance(m["content"],str) for m in messages)):
            raise OllamaError()
        if (type(max_output_tokens) is not int or not 1<=max_output_tokens<=2048
                or type(max_output_characters) is not int or not 1<=max_output_characters<=16000
                or type(timeout_seconds) not in (int,float) or not 0<timeout_seconds<=180):
            raise OllamaError()
        if sum(len(m['content'].encode('utf-8')) for m in messages)>120_000: raise OllamaError("context_overflow")
        started=time.monotonic(); deadline=started+timeout_seconds
        def live():
            if time.monotonic()>=deadline or cancelled is not None and cancelled.is_set(): raise OllamaError()
        live();before=self.identity();live()
        payload={"model":self.model,"messages":messages,"think":False,"truncate":False,"shift":False,
                 "options":{"num_ctx":self.context_tokens,"num_predict":max_output_tokens,"temperature":0,"seed":0},
                 "keep_alive":"5m"}
        if output_schema is not None: payload["format"]=output_schema
        rendered=self._request("POST","/api/chat",json={**payload,"stream":False,"_debug_render_only":True},
            timeout=httpx.Timeout(min(30,max(.1,deadline-time.monotonic()))))
        live()
        debug=rendered.get("_debug_info")
        if (rendered.get("model")!=self.model or rendered.get("remote_host") or rendered.get("remote_model")
                or not isinstance(debug,dict) or debug.get("image_count",0)!=0):
            raise OllamaError("model_unavailable")
        prompt=debug.get("rendered_template")
        # Fail closed if the runtime ignores render-only. Count the complete
        # rendered UTF-8 bytes, a deliberately conservative byte-token bound.
        if not isinstance(prompt,str) or not prompt: raise OllamaError("model_unavailable")
        if len(prompt.encode("utf-8"))+max_output_tokens+32>self.context_tokens:
            raise OllamaError("context_overflow")
        text=""; total=0; pending=bytearray(); final=None
        try:
            with self.client.stream("POST","/api/chat",json={**payload,"stream":True},
                    timeout=httpx.Timeout(min(10,max(.1,deadline-time.monotonic())),read=min(5,max(.1,deadline-time.monotonic())))) as response:
                if response.status_code!=200: raise OllamaError()
                for chunk in response.iter_bytes(chunk_size=None):
                    live();total+=len(chunk)
                    if total>1_000_000 or len(pending)+len(chunk)>100_000: raise OllamaError("output_limit")
                    pending.extend(chunk)
                    while b"\n" in pending:
                        line,_,rest=pending.partition(b"\n");pending=bytearray(rest)
                        if not line.strip(): continue
                        try: item=json.loads(line)
                        except (ValueError,UnicodeError): raise OllamaError() from None
                        if not isinstance(item,dict) or item.get("error") or item.get("remote_host") or item.get("remote_model"):
                            raise OllamaError()
                        if item.get("model")!=self.model: raise OllamaError("model_unavailable")
                        message=item.get("message",{})
                        if not isinstance(message,dict) or message.get("tool_calls") or message.get("thinking"):
                            raise OllamaError()
                        part=message.get("content","")
                        if not isinstance(part,str): raise OllamaError()
                        text+=part
                        if len(text.encode('utf-16-le'))//2>max_output_characters: raise OllamaError("output_limit")
                        if on_progress: on_progress(text)
                        if item.get("done") is True:
                            final=item
                            break
                    if final is not None: break
        except httpx.HTTPError: raise OllamaError() from None
        live()
        if not final or final.get("done_reason")!="stop" or not text.strip():
            raise OllamaError("output_limit" if final and final.get("done_reason")=="length" else "generation_failed")
        input_tokens=final.get("prompt_eval_count"); output_tokens=final.get("eval_count")
        if (type(input_tokens) is not int or type(output_tokens) is not int or input_tokens<1 or output_tokens<1
                or output_tokens>max_output_tokens or input_tokens+output_tokens>self.context_tokens):
            raise OllamaError("output_limit")
        if self.identity()!=before: raise OllamaError("model_unavailable")
        live()
        return {"text":text,"finishReason":"stop","inputTokens":input_tokens,"outputTokens":output_tokens}
