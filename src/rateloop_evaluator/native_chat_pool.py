"""Server-owned native Chat queue, with exact case leases and no training retention.

The transport credential permits only the application's persisted native Chat
bindings. Each content lease separately authorizes one workspace/case/model/rubric.
Existing local service inference and score semantics are reused unchanged.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time

import httpx

from .backends import GLINER_SCORE_CAPABILITY, tokenizer_commitment
from .connector import ConnectorRejected, ConnectorUnavailable, _hash, _opaque, _timestamp
from .hosted import validate_pinned_model
from .learning import LearningStore, provision_key
from .protocol import EvaluationRequest, EvaluationResult
from .service import Principal, create_app
from .storage import RuntimeStore
from .templates import CUSTOM_TEXT_CAPABILITY, bundle_supports_template, custom_text_seed, website_binary_question

WORKER_ID = "rateloop-native-chat-v1"
ENDPOINT = "/api/internal/native-chat-evaluator"


def validate_registrations(bundles: list[dict], model_dir: str) -> list[dict]:
    """Bind advertised registrations to the actual reviewed public volume weights."""
    model = validate_pinned_model(Path(model_dir))
    if len(bundles) != 2 or {bundle.get("language") for bundle in bundles} != {"en", "de"}:
        raise ValueError("Native Chat requires exact English and German registrations")
    for bundle in bundles:
        language = bundle["language"]
        if (bundle.get("baseWeightsCommitment") != "sha256:"+model["files"]["model.safetensors"]
                or bundle.get("tokenizerCommitment") != tokenizer_commitment(model)
                or bundle.get("adapterCommitment") is not None or bundle.get("trainingSnapshotCommitment") is not None
                or bundle.get("taskCapability") != CUSTOM_TEXT_CAPABILITY
                or bundle.get("scoreCapability") != GLINER_SCORE_CAPABILITY
                or bundle.get("maxTokens") != 512
                or bundle.get("template") != custom_text_seed(language).model_dump()
                or any(criterion.get("calibrationId") is not None for criterion in bundle.get("criteria", []))):
            raise PermissionError("Native Chat registration differs from the reviewed public model")
        _opaque(bundle.get("modelBundleId")); _hash(bundle.get("templateCommitment"))
    return deepcopy(bundles)


def evaluate_native_job(job: dict, *, backend, now=None) -> dict:
    """A real, ephemeral, single-case use grant; no raw input is written to disk."""
    current = time.time() if now is None else now
    request = EvaluationRequest.model_validate(job.get("content", {}).get("request"))
    content = job["content"]
    expires = _timestamp(job.get("leaseExpiresAt"))
    if not current < expires <= current+125:
        raise PermissionError("Native Chat case lease expired or exceeds its maximum")
    if (job.get("workerId") != WORKER_ID or request.workspaceId != job.get("workspaceId")
            or request.modelBundleId != job.get("modelBundleId")
            or request.input_commitment() != job.get("inputCommitment")
            or request.template_commitment() != job.get("templateCommitment")
            or (content.get("agentId"), content.get("agentVersionId")) != (job.get("agentId"), job.get("agentVersionId"))):
        raise PermissionError("Native Chat content changed the exact leased identity")
    website_binary_question(request.template)
    if content.get("retainForTraining") is not False:
        raise PermissionError("Native Chat pool never accepts training retention")
    registration = job.get("baseRegistration", {})
    if (registration.get("modelBundleId") != request.modelBundleId or registration.get("language") != request.template.language
            or registration.get("taskCapability") != CUSTOM_TEXT_CAPABILITY
            or registration.get("adapterCommitment") is not None or registration.get("trainingSnapshotCommitment") is not None):
        raise PermissionError("Native Chat model registration changed")
    bundle = {"id": request.modelBundleId, "languages": [registration["language"]],
        "template_commitments": [registration["templateCommitment"]], "task_capability": CUSTOM_TEXT_CAPABILITY,
        "max_tokens": 512, "calibrations": []}
    if not bundle_supports_template(bundle, request.template):
        raise PermissionError("Unsupported native Chat rubric")
    with tempfile.TemporaryDirectory(prefix="rateloop-native-case-") as temporary:
        root = Path(temporary); key = root/"key"; provision_key(key)
        learning = LearningStore(root/"learning", key)
        learning.add_grant(workspace_id=request.workspaceId, rights=["ai_use"], expires_at=expires,
            authorization_until=expires, case_ids=[request.caseId], template_ids=[request.template.id],
            fields=["input.text", "input.context"], model_bundle_ids=[request.modelBundleId],
            template_commitments=[request.template_commitment()], evidence="native-case-lease:"+job["jobId"])
        runtime = RuntimeStore(root/"runtime.sqlite", key)
        identity = Principal(request.workspaceId, frozenset({"evaluate"}))
        app = create_app(backend=backend, bundle=bundle, learning=learning, runtime=runtime,
            tokens={"0"*64: identity}, validate_bundle=lambda value: {"mode": "shadow"},
            allow_training_retention=lambda value: False)
        return app.state.evaluate(request, identity)


class NativeChatPool:
    def __init__(self, *, secret: str, base_url: str, bundles: list[dict], backend, transport=None):
        if base_url != "https://www.rateloop.ai" or not isinstance(secret, str) or not 32 <= len(secret) <= 256 or any(c in secret for c in "\r\n"):
            raise ValueError("Native Chat requires the canonical origin and dedicated server credential")
        self.bundles = deepcopy(bundles); self.backend = backend; self.pending = None
        self.client = httpx.Client(base_url=base_url, headers={"Authorization": "Bearer "+secret},
            timeout=20, follow_redirects=False, trust_env=False, transport=transport)

    def close(self):
        self.client.close(); self.pending = None

    def _request(self, body: dict) -> dict:
        try:
            with self.client.stream("POST", ENDPOINT, json=body) as response:
                if response.status_code >= 500 or response.status_code in (408,425,429):
                    raise ConnectorUnavailable("Native Chat queue is temporarily unavailable")
                if not 200 <= response.status_code < 300:
                    raise ConnectorRejected(response.status_code)
                payload = bytearray()
                for chunk in response.iter_bytes(chunk_size=4096):
                    if len(payload)+len(chunk) > 128_000: raise ValueError("Native Chat response exceeds the bounded limit")
                    payload.extend(chunk)
                value = json.loads(payload)
                if not isinstance(value, dict): raise ValueError("Native Chat response must be an object")
                return value
        except httpx.TransportError as error:
            raise ConnectorUnavailable("Native Chat queue is unreachable") from error

    def _submit_pending(self):
        job,result=self.pending
        # Offline receipt/failure retries never extend the original case grant.
        if _timestamp(job["leaseExpiresAt"]) <= time.time():
            self.pending=None
            return {"state":"lease_lost"}
        failing=job.get("failureCode") is not None
        body={"action":"fail" if failing else "complete","workspaceId":job["workspaceId"],
            "jobId":job["jobId"],"leaseToken":job["leaseToken"]}
        body.update({"retryable":False,"errorCode":job["failureCode"]} if failing else {"result":result})
        try:
            self._request(body)
        except ConnectorRejected as error:
            if failing or error.status in (401,403,404,409,410):
                self.pending=None
                return {"state":"lease_lost"}
            # A rejected receipt must not block every other tenant. Discard its
            # result and retry only the content-free failure intent if offline.
            self.pending=({**job,"failureCode":"native_receipt_rejected"},None)
            return self._submit_pending()
        self.pending=None
        return {"state":"failed" if failing else "completed"}

    def run_once(self):
        # A process restart releases this bounded in-memory receipt/failure;
        # the durable server lease and attempt ceiling recover an unfinished job.
        if self.pending: return self._submit_pending()
        job = self._request({"action": "claim", "bundles": self.bundles}).get("job")
        if job is None: return {"state": "idle"}
        if not isinstance(job, dict): raise ValueError("Native Chat claim must be an object")
        base = next((bundle for bundle in self.bundles if bundle["modelBundleId"] == job.get("modelBundleId")), None)
        if base is None or job.get("baseRegistration") != base:
            raise PermissionError("Native Chat queue requested an unreviewed model")
        binding = {"workspaceId": job.get("workspaceId"), "jobId": job.get("jobId"), "leaseToken": job.get("leaseToken")}
        try:
            lease = self._request({"action": "heartbeat_job", **binding})
            job["leaseExpiresAt"] = lease["leaseExpiresAt"]
            result = EvaluationResult.model_validate(evaluate_native_job(job, backend=self.backend)).model_dump()
            # Drop answer/context immediately after inference, even if transport is offline.
            self.pending = ({"workspaceId": job["workspaceId"], "jobId": job["jobId"], "leaseToken": job["leaseToken"],
                "leaseExpiresAt":job["leaseExpiresAt"]}, result)
            return self._submit_pending()
        except ConnectorRejected as error:
            if error.status not in (401,403,404,409,410): raise
            return {"state": "lease_lost"}
        except (ValueError, PermissionError):
            try: self._request({"action": "fail", **binding, "retryable": False, "errorCode": "native_validation_failed"})
            except (ConnectorUnavailable,ConnectorRejected): pass
            return {"state": "failed"}
