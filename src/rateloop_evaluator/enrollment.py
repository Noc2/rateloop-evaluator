"""Pair one customer-owned worker without exposing a local listening port."""
from __future__ import annotations

from argparse import Namespace
import hashlib
import ipaddress
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

import httpx

from .connector import bounded_response_object, _opaque
from .learning import read_secret

_ENROLL = "/api/assurance/v2/evaluations/enroll"
_IDENTITY = ("workspaceId", "deviceId", "workerId", "agentId", "agentVersionId")
_UNCERTAIN = "Pairing response was lost. Revoke this device in RateLoop and create a new pairing code before trying again."


def enrollment_origin(value: str, *, allow_insecure_loopback: bool = False) -> str:
    url = urlsplit(value)
    if not url.hostname or url.username or url.password or url.query or url.fragment or url.path not in ("", "/"):
        raise ValueError("RateLoop address must be an origin without a path or credentials")
    loopback = url.hostname == "localhost"
    try: loopback = loopback or ipaddress.ip_address(url.hostname).is_loopback
    except ValueError: pass
    if url.scheme != "https" and not (url.scheme == "http" and loopback and allow_insecure_loopback):
        raise ValueError("Pairing requires HTTPS")
    return value.rstrip("/")


def _identity(value: dict) -> dict:
    identity = {key: _opaque(value.get(key)) for key in _IDENTITY}
    capabilities = value.get("capabilities")
    if not isinstance(capabilities, list) or not capabilities or len(set(capabilities)) != len(capabilities) or set(capabilities) - {"evaluation", "generation"}:
        raise ValueError("Pairing must permit a supported capability on this device")
    identity["capabilities"] = capabilities
    return identity


def connect(*, state_dir: str | Path, base_url: str, enrollment_token: str, model_dir: str | Path | None = None,
            generation_model: str | None = None, ollama_url: str = "http://127.0.0.1:11434", context_tokens: int = 8192,
            device: str = "cpu", allow_insecure_loopback: bool = False,
            transport: httpx.BaseTransport | None = None) -> dict:
    """Preview identity, prepare local manifests, then consume the code once.

    No grants are created. A lost consuming response requires owner revocation:
    replaying it could otherwise leave an unknown live device credential.
    """
    from . import cli
    from .hosted import bootstrap, validate_pinned_model
    from .templates import CUSTOM_TEXT_CAPABILITY
    root = Path(state_dir).expanduser().absolute()
    model = Path(model_dir).expanduser().resolve(strict=True) if model_dir else None
    if root.is_symlink() or root.resolve() != root or model is not None and (root == model or model.is_relative_to(root)):
        raise ValueError("Use separate absolute, non-symlink state and model directories")
    if device not in ("cpu", "mps", "cuda"):
        raise ValueError("Unsupported local device")
    origin = enrollment_origin(base_url, allow_insecure_loopback=allow_insecure_loopback)
    if not isinstance(enrollment_token, str) or not re.fullmatch(r"[A-Za-z0-9_-]{20,256}", enrollment_token):
        raise ValueError("Pairing code is invalid")
    if root.exists() and any(root.iterdir()):
        raise ValueError("Pairing requires a new state directory; keep existing worker credentials separate")
    if model is None and generation_model is None:
        raise ValueError("Select an existing rating model directory or a local generation model")
    if model is not None: validate_pinned_model(model)
    generation = None
    if generation_model is not None:
        from .ollama import OllamaRuntime
        runtime = OllamaRuntime(model=generation_model, base_url=ollama_url, context_tokens=context_tokens)
        try: generation = {"baseUrl": ollama_url, "model": runtime.identity()}
        finally: runtime.close()
    with httpx.Client(base_url=origin, timeout=30, trust_env=False, follow_redirects=False, transport=transport) as client:
        try:
            with client.stream("POST", _ENROLL, json={"enrollmentToken": enrollment_token}) as response:
                if response.status_code != 200:
                    raise ValueError("Pairing code was rejected or expired; create a new code in RateLoop")
                preview = _identity(bounded_response_object(response, max_bytes=16_384))
        except httpx.HTTPError:
            raise RuntimeError("RateLoop could not be reached; pairing has not been claimed") from None
        if model is not None and "evaluation" not in preview["capabilities"] or generation is not None and "generation" not in preview["capabilities"]:
            raise ValueError("Pairing does not permit the locally selected capability")
        seed = hashlib.sha256((preview["deviceId"] + ":gliner25").encode()).hexdigest()[:24]
        bundles = [{"language": language, "modelBundleId": "local-" + seed + "-" + language}
                   for language in ("en", "de")]
        bundles += [{"language": language, "modelBundleId": "local-" + seed + "-custom-" + language,
                     "taskCapability": dict(CUSTOM_TEXT_CAPABILITY)} for language in ("en", "de")]
        connection = {"baseUrl": origin, "apiKey": "pending-enrollment", "apiKeyId": "pending-enrollment",
                      "agentId": preview["agentId"], "agentVersionId": preview["agentVersionId"], "metadataUploadEnabled": True}
        config = {"workspaceId": preview["workspaceId"], "workerId": preview["workerId"], "modelDir": str(model),
                  "stateDir": str(root), "bundles": bundles, "connection": connection}
        if model is not None:
            prepared = bootstrap(config)
            registrations = [json.loads(read_secret(path)) for path in prepared["registrations"]]
        else:
            bundles = []
            cli.run(Namespace(command="init", state_dir=str(root), workspace=preview["workspaceId"]))
            registrations = []
        cli.write_private(root / "enrollment.json", {**preview, "baseUrl": origin, "status": "claiming"})
        # Exactly one consuming request. Do not retry this request automatically.
        try:
            with client.stream("POST", _ENROLL, json={"enrollmentToken": enrollment_token, "bundles": registrations}) as response:
                if response.status_code != 200:
                    raise RuntimeError("Pairing was not confirmed. Revoke this device in RateLoop and create a new pairing code.")
                claimed = bounded_response_object(response, max_bytes=32_768)
            if _identity(claimed) != preview:
                raise ValueError("Pairing identity changed")
            expected = [bundle["modelBundleId"] for bundle in bundles]
            if claimed.get("modelBundleIds") != expected:
                raise ValueError("Pairing model identities changed")
            key = claimed.get("apiKey")
            if not isinstance(key, str) or not 20 <= len(key) <= 512 or any(c in key for c in "\r\n"):
                raise ValueError("Invalid worker credential")
            connection.update(apiKey=key, apiKeyId=_opaque(claimed.get("apiKeyId")))
            if allow_insecure_loopback: connection["allowInsecureLoopback"] = True
            cli.write_private(root / "connector.json", connection)
            cli.write_private(root / "worker.json", {"schemaVersion": "rateloop.local-worker.v1", "workerId": preview["workerId"],
                "modelBundleIds": expected, "device": device, "pollSeconds": 5, **({"generation": generation} if generation else {})})
            cli.write_private(root / "enrollment.json", {**preview, "baseUrl": origin, "status": "connected"})
        except (httpx.HTTPError, ValueError, KeyError, OSError, RuntimeError):
            raise RuntimeError(_UNCERTAIN) from None
    return {"status": "connected", "stateDirectory": str(root), "deviceId": preview["deviceId"],
            "modelBundleIds": expected, "processingEnabled": False,
            "next": "Run rateloop-evaluator start, then select this device for AI ratings in RateLoop."}


def worker_arguments(state_dir: str | Path, *, command: str = "worker", once: bool = False,
                     load: bool = False, output: str | None = None) -> Namespace:
    root = Path(state_dir).expanduser().resolve(strict=True)
    record = json.loads(read_secret(root / "worker.json"))
    if (not isinstance(record, dict) or set(record) - {"schemaVersion", "workerId", "modelBundleIds", "device", "pollSeconds", "generation"}
            or not {"schemaVersion", "workerId", "modelBundleIds", "device", "pollSeconds"} <= set(record)
            or record["schemaVersion"] != "rateloop.local-worker.v1"):
        raise ValueError("Invalid paired worker configuration")
    _opaque(record["workerId"])
    bundles = record["modelBundleIds"]
    if not isinstance(bundles, list) or not (0 if record.get("generation") else 1) <= len(bundles) <= 20 or len(set(bundles)) != len(bundles):
        raise ValueError("Invalid paired model allowlist")
    for bundle in bundles: _opaque(bundle)
    if record["device"] not in ("cpu", "mps", "cuda") or type(record["pollSeconds"]) not in (int, float) or not 1 <= record["pollSeconds"] <= 60:
        raise ValueError("Invalid paired worker settings")
    return Namespace(generation=record.get("generation"), command=command, state_dir=str(root), config=str(root / "connector.json"), worker_id=record["workerId"],
        bundle_id=bundles, device=record["device"], poll_seconds=record["pollSeconds"], allow_training=False,
        training_model_dir=None, once=once, load=load,
        output=output or str(Path.home() / "Library/LaunchAgents" / ("ai.rateloop.evaluator." + hashlib.sha256(record["workerId"].encode()).hexdigest()[:16] + ".plist")))
