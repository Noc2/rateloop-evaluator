"""Explicit outbound RateLoop metadata connector; never a raw-content transport.

HTTPS and an API-key recipient bind mirrored grants. Offline grants expire at the
server's original lease deadline (at most 24 hours); unseen remote revocations are
not instant offline. Audit blinding is connector-attested, not cryptographic proof.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
import re
import time
from typing import Any, Callable
from urllib.parse import urlsplit

import httpx

from .learning import LearningStore, _digest
from .protocol import EvaluationRequest, EvaluationResult, commitment
from .storage import RuntimeStore

_API = "/api/assurance/v2/evaluations"
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$")
_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_GRANT_KEYS = {"grantId", "workspaceId", "apiKeyId", "purpose", "modelBundleId", "templateCommitment", "fields",
               "publicWeightsAllowed", "issuedAt", "expiresAt", "revokedAt", "revision"}
_FIELD_MAP = {"input":"input.text", "context":"input.context", "evidence":"input.evidence", "human_labels":"human_labels"}
_CONSENT_KEYS = {"consentId","revision","workspaceId","apiKeyId","purpose","processingLocation","modelFamilyId",
                 "modelBundleIds","templateCommitments","fields","issuedAt","expiresAt","revokedAt"}
_PURPOSE_RIGHTS = {"ai_use":"ai_use","private_learning":"private_training","shared_contribution":"shared_contribution",
                   "public_weights":"public_weight_distribution"}


class ConnectorUnavailable(RuntimeError):
    """No fresh server result; offline work must not claim an independent audit."""

    def __init__(self, message: str, *, coordination_busy: bool = False):
        self.coordination_busy = coordination_busy
        super().__init__(message)


class AuthorizationLeaseRejected(PermissionError):
    """Fixed diagnostic category; no remote values, identifiers or content."""

    def __init__(self, reason: str):
        if reason not in {"workspace_mismatch","recipient_mismatch","watermark_mismatch",
                          "invalid_issue_time","future_issue","expired","excessive_duration"}:
            raise ValueError("Unknown authorization lease rejection reason")
        self.reason = reason
        super().__init__("Worker authorization lease rejected: "+reason)


def _coordination_busy(response: httpx.Response) -> bool:
    """Classify one bounded, known retry response without exposing its body."""
    if response.status_code != 503:
        return False
    payload = bytearray()
    for chunk in response.iter_bytes(chunk_size=4096):
        if len(payload) + len(chunk) > 4096:
            return False
        payload.extend(chunk)
    try:
        value = json.loads(payload)
    except (ValueError, UnicodeError):
        return False
    return (isinstance(value, dict) and value.get("code") == "database_coordination_busy"
            and value.get("retryable") is True)


class ConnectorRejected(ValueError):
    def __init__(self, status: int):
        self.status = status
        super().__init__(f"RateLoop rejected the connector request (HTTP {status})")


def _timestamp(value: Any) -> float:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("Expected a UTC timestamp")
    try:
        return datetime.fromisoformat(value[:-1]+"+00:00").timestamp()
    except (ValueError, OverflowError) as exc:
        raise ValueError("Invalid timestamp") from exc


def _opaque(value: Any) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError("Expected an opaque metadata identifier")
    return value


def _hash(value: Any) -> str:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise ValueError("Expected a content commitment")
    return value


def _review_context(value: dict) -> dict:
    """Allow only execution metadata; never accept review text or free-form maps."""
    keys = {"policyId", "policyVersion", "workflowKey", "riskTier", "audiencePolicyHash", "declaredConfidenceBps", "metadataComplete", "execution"}
    if not isinstance(value, dict) or set(value) - keys or not keys - {"declaredConfidenceBps"} <= set(value):
        raise ValueError("Review context accepts only the documented metadata fields")
    for key in ("policyId", "workflowKey", "riskTier"):
        _opaque(value[key])
    _hash(value["audiencePolicyHash"])
    if type(value["metadataComplete"]) is not bool or type(value["policyVersion"]) is not int or value["policyVersion"] < 1:
        raise ValueError("Invalid review-policy metadata")
    bps=value.get("declaredConfidenceBps")
    if bps is not None and (type(bps) is not int or not 0 <= bps <= 10000):
        raise ValueError("Invalid declared confidence")
    execution=value["execution"]
    execution_keys={"externalExecutionId","status","startedAt","completedAt","toolCallCount","toolDurationMs","primarySpanId","generationSpans"}
    if not isinstance(execution,dict) or set(execution)-execution_keys or not {"externalExecutionId","status","primarySpanId","generationSpans"} <= set(execution):
        raise ValueError("Execution accepts only documented metadata fields")
    _opaque(execution["externalExecutionId"]); _opaque(execution["primarySpanId"])
    if execution["status"] not in ("completed","failed"):
        raise ValueError("Invalid execution status")
    spans=execution["generationSpans"]
    if not isinstance(spans,list) or not 1 <= len(spans) <= 64:
        raise ValueError("Execution requires 1-64 bounded generation spans")
    span_keys={"spanId","parentSpanId","role","provider","requestedModel","resolvedModel","modelVersion","reasoningEffort","serviceTier",
               "startedAt","completedAt","timeToFirstOutputMs","inputTokens","cachedInputTokens","outputTokens","reasoningOutputTokens","responseIdHash","finishReason"}
    for item in [execution,*spans]:
        if not isinstance(item,dict) or (item is not execution and (set(item)-span_keys or not {"spanId","role","provider","requestedModel"} <= set(item))):
            raise ValueError("Generation span contains unsupported metadata")
        for key,field in item.items():
            if key == "generationSpans" or field is None:
                continue
            if key in ("startedAt","completedAt"):
                _timestamp(field)
            elif key.endswith("Tokens") or key.endswith("Ms") or key == "toolCallCount":
                if type(field) is not int or not 0 <= field <= 2**31-1:
                    raise ValueError("Invalid execution count")
            elif key == "responseIdHash":
                _hash(field)
            elif not isinstance(field,str) or not re.fullmatch(r"[A-Za-z0-9._:/-]{1,200}",field):
                raise ValueError("Execution metadata must contain identifiers, never free-form content")
    return deepcopy(value)


class RateLoopConnector:
    def __init__(self, *, base_url: str, api_key: str, api_key_id: str, workspace_id: str,
                 agent_id: str, agent_version_id: str, learning: LearningStore, runtime: RuntimeStore,
                 metadata_upload_enabled: bool = False, allow_insecure_loopback: bool = False,
                 transport: httpx.BaseTransport | None = None):
        url=urlsplit(base_url)
        if not url.hostname or url.username or url.password or url.query or url.fragment or url.path not in ("","/"):
            raise ValueError("Connector destination must be an explicit origin without credentials or paths")
        loopback=url.hostname == "localhost"
        try:
            loopback=loopback or ipaddress.ip_address(url.hostname).is_loopback
        except ValueError:
            pass
        if url.scheme != "https" and not (url.scheme == "http" and loopback and allow_insecure_loopback):
            raise ValueError("HTTPS is required; HTTP needs explicit loopback-only development permission")
        if not api_key or any(c in api_key for c in "\r\n"):
            raise ValueError("A valid API credential is required")
        for value in (api_key_id,workspace_id,agent_id,agent_version_id):
            _opaque(value)
        if type(metadata_upload_enabled) is not bool:
            raise ValueError("Metadata upload permission must be explicit")
        self.base_url=base_url.rstrip("/")
        self.workspace_id,self.api_key_id=workspace_id,api_key_id
        self.agent_id,self.agent_version_id=agent_id,agent_version_id
        self.learning,self.runtime=learning,runtime
        self.metadata_upload_enabled=metadata_upload_enabled
        self.namespace=hashlib.sha256(json.dumps([self.base_url,workspace_id,api_key_id]).encode()).hexdigest()
        self.client=httpx.Client(base_url=self.base_url,headers={"Authorization":"Bearer "+api_key},
                                 timeout=15,trust_env=False,follow_redirects=False,transport=transport)

    def close(self) -> None:
        self.client.close()

    def _state(self, database: dict) -> dict:
        state=database.setdefault("connectors",{}).setdefault(self.namespace,
            {"watermark":-1,"grants":{},"audits":{},"results":{},"imports":{},"dead_letters":{},"mode":"off"})
        state["workspace_id"]=self.workspace_id
        return state

    def _case_live(self, database: dict, case_id: str) -> None:
        if _digest([self.workspace_id,case_id]) in database.get("deleted_cases",{}):
            raise PermissionError("This case was deleted; connector reingestion is not allowed")

    def _request(self, method: str, endpoint: str, **kwargs) -> dict:
        try:
            with self.client.stream(method,_API+endpoint,**kwargs) as response:
                if response.status_code in (401,403):
                    self._revoke_mirrors("remote_credential_rejected")
                    raise PermissionError("RateLoop credential or scope is no longer authorized")
                if response.status_code >= 500 or response.status_code == 429:
                    raise ConnectorUnavailable("RateLoop temporarily unavailable; retry the persisted receipt",
                        coordination_busy=_coordination_busy(response))
                if not 200 <= response.status_code < 300:
                    raise ConnectorRejected(response.status_code)
                declared=response.headers.get("content-length")
                if declared is not None and (not declared.isdigit() or int(declared)>10_000_000):
                    raise ValueError("RateLoop response exceeds the bounded metadata limit")
                payload=bytearray()
                # iter_bytes counts decompressed data too, bounding compressed
                # responses before they can become an unbounded JSON document.
                for chunk in response.iter_bytes(chunk_size=65536):
                    if len(payload)+len(chunk)>10_000_000:
                        raise ValueError("RateLoop response exceeds the bounded metadata limit")
                    payload.extend(chunk)
                value=json.loads(payload)
        except httpx.TransportError as exc:
            raise ConnectorUnavailable("RateLoop is unavailable; no fresh audit or revocation claim is available") from exc
        if not isinstance(value,dict):
            raise ValueError("RateLoop response must be an object")
        return value

    def _revoke_mirrors(self, reason: str) -> None:
        with self.learning.transaction() as database:
            state=self._state(database)
            ids=[r["local_id"] for r in [*state["grants"].values(),*state.get("consents",{}).values()]]
            state["last_failure"]=reason
        for grant_id in ids:
            self.learning.revoke_grant(grant_id,self.workspace_id)

    def _validate_grant(self, grant: dict, *, now: float, watermark: int) -> dict:
        if not isinstance(grant,dict) or set(grant) != _GRANT_KEYS:
            raise ValueError("Incomplete or unexpected remote grant fields")
        if grant["workspaceId"] != self.workspace_id or grant["apiKeyId"] != self.api_key_id:
            raise PermissionError("Remote grant recipient does not match this connector")
        for key in ("grantId","modelBundleId"):
            _opaque(grant[key])
        _hash(grant["templateCommitment"])
        if grant["purpose"] not in ("private_learning","shared_contribution") or type(grant["publicWeightsAllowed"]) is not bool:
            raise ValueError("Invalid learning purpose")
        fields=grant["fields"]
        if not isinstance(fields,list) or not fields or len(fields)!=len(set(fields)) or not set(fields) <= set(_FIELD_MAP):
            raise ValueError("Invalid remote field scope")
        issued,expires=_timestamp(grant["issuedAt"]),_timestamp(grant["expiresAt"])
        if not 0 < issued <= now or not 0 < expires-issued <= 86400:
            raise ValueError("Remote grant has an invalid issue time or lease duration")
        if type(grant["revision"]) is not int or not 0 <= grant["revision"] <= watermark:
            raise ValueError("Remote grant is newer than its revocation watermark")
        if grant["revokedAt"] is not None:
            revoked=_timestamp(grant["revokedAt"])
            if not issued <= revoked <= now:
                raise ValueError("Invalid remote revocation timestamp")
        return deepcopy(grant)

    def _sync_deletions(self, response: dict) -> None:
        """Apply explicit workspace case-erasure tombstones, never infer deletion from auth failures."""
        if "deletedCases" not in response: return
        deleted=response["deletedCases"]; watermark=response.get("deletionWatermark")
        if not isinstance(deleted,list) or len(deleted)>100000 or type(watermark) is not int or watermark<0:
            raise ValueError("Invalid deletion synchronization")
        with self.learning.transaction() as database:
            previous=self._state(database).get("deletion_watermark",0)
        if watermark<previous: raise PermissionError("Deletion watermark moved backwards")
        for item in deleted:
            if not isinstance(item,dict) or set(item)!={"caseId","deletedAt"}:
                raise ValueError("Invalid deleted-case tombstone")
            _opaque(item["caseId"]); _timestamp(item["deletedAt"])
        for item in deleted:
            self.learning.delete_case(self.workspace_id,item["caseId"])
            self.runtime.delete_case(self.workspace_id,item["caseId"])
        with self.learning.transaction() as database:
            self._state(database)["deletion_watermark"]=watermark

    def _validate_consent(self, consent: dict, *, now: float, watermark: int) -> dict:
        if not isinstance(consent,dict) or set(consent) != _CONSENT_KEYS:
            raise ValueError("Incomplete or unexpected durable consent fields")
        if consent["workspaceId"] != self.workspace_id or consent["apiKeyId"] != self.api_key_id:
            raise PermissionError("Durable consent recipient mismatch")
        for key in ("consentId","modelFamilyId","processingLocation"):
            _opaque(consent[key])
        if consent["purpose"] not in _PURPOSE_RIGHTS:
            raise ValueError("Invalid durable consent purpose")
        for key,validate in (("modelBundleIds",_opaque),("templateCommitments",_hash)):
            if not isinstance(consent[key],list) or not 1 <= len(consent[key]) <= 100 or len(set(consent[key])) != len(consent[key]):
                raise ValueError("Consent requires explicit bounded model and template scopes")
            for value in consent[key]: validate(value)
        if not isinstance(consent["fields"],list) or not consent["fields"] or not set(consent["fields"]) <= set(_FIELD_MAP):
            raise ValueError("Invalid consent fields")
        issued=_timestamp(consent["issuedAt"])
        if not 0 < issued <= now or type(consent["revision"]) is not int or not 1 <= consent["revision"] <= watermark:
            raise ValueError("Invalid consent version or issue time")
        if consent["expiresAt"] is not None and _timestamp(consent["expiresAt"]) <= issued:
            raise ValueError("Invalid durable consent expiration")
        if consent["revokedAt"] is not None and not issued <= _timestamp(consent["revokedAt"]) <= now:
            raise ValueError("Invalid durable consent revocation")
        return deepcopy(consent)

    def _sync_consents(self, response: dict, *, now: float, watermark: int) -> int:
        consents=response.get("consents",[])
        if not isinstance(consents,list) or len(consents)>1000:
            raise ValueError("Invalid durable consent collection")
        consents=[self._validate_consent(c,now=now,watermark=watermark) for c in consents]
        if len({c["consentId"] for c in consents}) != len(consents):
            raise ValueError("Duplicate consent identity")
        lease=response.get("authorizationLease")
        until=now
        if consents:
            expected={"leaseId","issuedAt","expiresAt","revocationWatermark","recipientApiKeyId","workspaceId"}
            if not isinstance(lease,dict) or set(lease) != expected:
                raise ValueError("Durable permissions require a scoped worker authorization lease")
            _opaque(lease["leaseId"])
            issued,until=_timestamp(lease["issuedAt"]),_timestamp(lease["expiresAt"])
            if lease["workspaceId"] != self.workspace_id:
                raise AuthorizationLeaseRejected("workspace_mismatch")
            if lease["recipientApiKeyId"] != self.api_key_id:
                raise AuthorizationLeaseRejected("recipient_mismatch")
            if lease["revocationWatermark"] != watermark:
                raise AuthorizationLeaseRejected("watermark_mismatch")
            if issued <= 0:
                raise AuthorizationLeaseRejected("invalid_issue_time")
            if issued > now:
                raise AuthorizationLeaseRejected("future_issue")
            if until <= now:
                raise AuthorizationLeaseRejected("expired")
            if until-issued>900:
                raise AuthorizationLeaseRejected("excessive_duration")
        with self.learning.transaction() as database:
            previous=deepcopy(self._state(database).get("consents",{}))
        received={c["consentId"]:c for c in consents}
        for identity,old in previous.items():
            current=received.get(identity)
            if (not current or current["revokedAt"] is not None
                    or (current["expiresAt"] is not None and _timestamp(current["expiresAt"])<=now)
                    or current["revision"] != old["consent"]["revision"]):
                self.learning.revoke_grant(old["local_id"],self.workspace_id,now=now)
            elif commitment(current,"rateloop.durable-consent.v1") != old["digest"]:
                raise PermissionError("Durable consent changed without a new explicit revision")
        mirrored={}
        for consent in consents:
            expiration=_timestamp(consent["expiresAt"]) if consent["expiresAt"] else 253402300799.0
            if consent["revokedAt"] is not None or expiration<=now:
                continue
            identity=consent["consentId"]
            local_id="consent_"+hashlib.sha256((self.namespace+identity+":"+str(consent["revision"])).encode()).hexdigest()[:48]
            old=previous.get(identity)
            if old is not None and old["local_id"] == local_id:
                self.learning.renew_authorization(local_id,self.workspace_id,min(until,expiration),now=now)
            else:
                self.learning.add_grant(workspace_id=self.workspace_id,rights=[_PURPOSE_RIGHTS[consent["purpose"]]],
                    expires_at=expiration,authorization_until=min(until,expiration),fields=[_FIELD_MAP[f] for f in consent["fields"]],
                    model_bundle_ids=consent["modelBundleIds"],template_commitments=consent["templateCommitments"],grant_id=local_id,
                    evidence="RateLoop durable consent "+identity+" revision "+str(consent["revision"]),now=now)
            mirrored[identity]={"local_id":local_id,"digest":commitment(consent,"rateloop.durable-consent.v1"),"consent":consent}
            with self.learning.transaction() as database:
                self._state(database).setdefault("consents",{})[identity]=mirrored[identity]
        with self.learning.transaction() as database:
            self._state(database).update(consents=mirrored,authorization_lease=lease)
        return len(mirrored)

    def sync_grants(self, *, now: float | None = None) -> dict:
        response=self._request("GET","/grants")
        # The server issues the lease during this request, after our send time.
        current=time.time() if now is None else now
        created_ids=[]
        try:
            if response.get("workspaceId") != self.workspace_id or response.get("recipientApiKeyId") != self.api_key_id:
                raise PermissionError("Grant response is not bound to the configured workspace and API-key recipient")
            if response.get("workspaceDeletion") is not None:
                notice=response["workspaceDeletion"]
                if not isinstance(notice,dict) or set(notice)!={"deletedAt"} or not 0<_timestamp(notice["deletedAt"])<=current+300:
                    raise ValueError("Invalid owner workspace-deletion notice")
                self._revoke_mirrors("owner_workspace_deleted")
                if response.get("deletedCases"): self._sync_deletions(response)
                with self.learning.transaction() as database:
                    self._state(database).update(mode="off",workspace_deleted_at=notice["deletedAt"])
                return {"mode":"off","workspaceDeleted":True,"mirroredGrants":0,"mirroredConsents":0,
                    "localErasure":"Explicit case tombstones or owner delete-case commands are required for remaining local data."}
            with self.learning.transaction() as database:
                if self._state(database).get("workspace_deleted_at") is not None:
                    raise PermissionError("An owner-deleted workspace cannot renew execution permissions")
            watermark=response.get("revocationWatermark")
            if type(watermark) is not int or watermark < 0 or not isinstance(response.get("grants"),list):
                raise ValueError("Invalid remote revocation watermark")
            mode=response.get("settings",{}).get("mode")
            if mode not in ("off","shadow","paused"):
                raise ValueError("Unsupported hosted evaluator mode")
            grants=[self._validate_grant(g,now=current,watermark=watermark) for g in response["grants"]]
            if len({g["grantId"] for g in grants}) != len(grants):
                raise ValueError("Duplicate remote grant identity")
            with self.learning.transaction() as database:
                previous=deepcopy(self._state(database))
            if watermark < previous["watermark"]:
                raise PermissionError("Remote revocation watermark moved backwards")
            self._sync_deletions(response)
            consent_count=self._sync_consents(response,now=current,watermark=watermark)
            received={g["grantId"]:g for g in grants}
            for remote_id,old in previous["grants"].items():
                remote=received.get(remote_id)
                if remote is None or remote["revokedAt"] is not None or _timestamp(remote["expiresAt"]) <= current:
                    self.learning.revoke_grant(old["local_id"],self.workspace_id,now=current)
                elif commitment(remote,"rateloop.remote-grant.v1") != old["digest"]:
                    raise PermissionError("An immutable grant changed; renewed permission needs a new grant identity")
            mirrored={}
            for grant in grants:
                if grant["revokedAt"] is not None or _timestamp(grant["expiresAt"]) <= current:
                    continue
                local_id="saas_"+hashlib.sha256((self.namespace+grant["grantId"]).encode()).hexdigest()[:48]
                digest=commitment(grant,"rateloop.remote-grant.v1")
                old=previous["grants"].get(grant["grantId"])
                if old is None:
                    rights=["private_training" if grant["purpose"]=="private_learning" else "shared_contribution"]
                    if grant["publicWeightsAllowed"]:
                        rights.append("public_weight_distribution")
                    self.learning.add_grant(workspace_id=self.workspace_id,rights=rights,expires_at=_timestamp(grant["expiresAt"]),
                        fields=[_FIELD_MAP[f] for f in grant["fields"]],model_bundle_ids=[grant["modelBundleId"]],
                        template_commitments=[grant["templateCommitment"]],grant_id=local_id,
                        evidence="RateLoop scoped grant "+grant["grantId"]+" at revision "+str(watermark),now=current)
                    created_ids.append(local_id)
                mirrored[grant["grantId"]]={"local_id":local_id,"digest":digest,"grant":grant}
            with self.learning.transaction() as database:
                state=self._state(database)
                if state["watermark"] > watermark:
                    raise PermissionError("A newer grant synchronization already completed")
                state.update({"watermark":watermark,"grants":mirrored,"synced_at":current,"mode":mode})
            return {"revocationWatermark":watermark,"mirroredGrants":len(mirrored),"mirroredConsents":consent_count,"mode":mode,
                    "offlineRevocation":"Original grant expiration bounds offline use; unseen revocations are not instant."}
        except (ValueError,PermissionError):
            for local_id in created_ids:
                self.learning.revoke_grant(local_id,self.workspace_id,now=current)
            self._revoke_mirrors("invalid_remote_grant_state")
            raise

    def _receipt(self, result: EvaluationResult | dict) -> dict:
        result=EvaluationResult.model_validate(result.model_dump() if isinstance(result,EvaluationResult) else result)
        if result.abstainReason is not None:
            _opaque(result.abstainReason)
        if result.workspaceId != self.workspace_id:
            raise PermissionError("Result belongs to another workspace")
        # This exact wire schema has no input, context, evidence or review text.
        return {"schemaVersion":"rateloop.automated-eval-receipt.v2","agentId":self.agent_id,
                "agentVersionId":self.agent_version_id,"result":result.model_dump()}

    def queue_result(self, result: EvaluationResult | dict, *, job_context: dict | None = None) -> str:
        if not self.metadata_upload_enabled:
            raise PermissionError("Receipt metadata upload has not been explicitly enabled")
        receipt=self._receipt(result)
        with self.learning.transaction() as database:
            self._case_live(database,receipt["result"]["caseId"])
        receipt_id="receipt_"+self.namespace[:16]+"_"+hashlib.sha256((self.namespace+receipt["result"]["resultCommitment"]).encode()).hexdigest()
        acknowledged=self.runtime.acknowledgment(receipt_id)
        if acknowledged is not None:
            if acknowledged.get("receiptHash") != commitment(receipt,"rateloop.product-evaluator.v2"):
                raise ValueError("Persisted receipt acknowledgment has a different commitment")
            return receipt_id
        if job_context is not None:
            if not isinstance(job_context,dict) or set(job_context)!={"jobId","workerId","leaseToken"}:
                raise ValueError("Job receipts require their exact execution fence")
            _opaque(job_context["jobId"]); _opaque(job_context["workerId"])
            if not isinstance(job_context["leaseToken"],str) or not 16<=len(job_context["leaseToken"])<=512 or any(c in job_context["leaseToken"] for c in "\r\n"):
                raise ValueError("Invalid receipt execution fence")
            with self.learning.transaction() as database:
                self._state(database).setdefault("receipt_jobs",{})[receipt_id]={**job_context,"caseId":receipt["result"]["caseId"]}
        self.runtime.enqueue(receipt_id,receipt)
        return receipt_id

    def flush(self, limit: int = 20, *, now: float | None = None) -> dict:
        if not self.metadata_upload_enabled:
            raise PermissionError("Receipt metadata upload has not been explicitly enabled")
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("Flush limit must be between 1 and 100")
        current=time.time() if now is None else now
        delivered,retried,rejected=0,0,0
        for receipt_id,receipt in self.runtime.pending(limit):
            if not receipt_id.startswith("receipt_"+self.namespace[:16]+"_"):
                continue
            try:
                if set(receipt) != {"schemaVersion","agentId","agentVersionId","result"} or receipt != self._receipt(receipt["result"]):
                    raise ValueError("Outbox payload is not this connector's exact metadata receipt")
                expected="receipt_"+self.namespace[:16]+"_"+hashlib.sha256((self.namespace+receipt["result"]["resultCommitment"]).encode()).hexdigest()
                if expected != receipt_id:
                    continue  # Another explicitly configured connector owns this row.
                observed=_timestamp(receipt["result"]["observedAt"])
                if not current-86400 <= observed <= current+300:
                    raise ValueError("Receipt expired outside the server's 24-hour ingest window")
                headers={"Idempotency-Key":receipt_id}
                with self.learning.transaction() as database:
                    job=self._state(database).get("receipt_jobs",{}).get(receipt_id)
                if job:
                    headers.update({"X-Evaluator-Job":job["jobId"],"X-Evaluator-Worker":job["workerId"],"X-Evaluator-Lease":job["leaseToken"]})
                response=self._request("POST","/receipts",json=receipt,headers=headers)
                if (response.get("schemaVersion") != "rateloop.automated-eval-ingest-result.v2" or response.get("outcome") not in (None,receipt["result"]["outcome"])
                    or response.get("receiptHash") != commitment(receipt,"rateloop.product-evaluator.v2")
                    or not re.fullmatch(r"aev_[0-9a-f]{40}",str(response.get("receiptId", "")))
                    or response.get("policy",{}).get("mayReduceHumanReview") is not False):
                    raise ValueError("Receipt acknowledgment does not match the submitted result")
                self.runtime.acknowledge(receipt_id,{"workspaceId":self.workspace_id,"caseId":receipt["result"]["caseId"],
                    "receiptId":response["receiptId"],"receiptHash":response["receiptHash"]}); delivered+=1
                with self.learning.transaction() as database:
                    self._state(database)["dead_letters"].pop(receipt_id,None)
            except ConnectorUnavailable:
                self.runtime.retry(receipt_id); retried+=1
            except PermissionError:
                self.runtime.retry(receipt_id)
                raise
            except (ValueError,KeyError,TypeError) as error:
                with self.learning.transaction() as database:
                    self._state(database)["dead_letters"][receipt_id]={"reason":str(error),"at":current}
                self.runtime.delivered(receipt_id); rejected+=1
        return {"delivered":delivered,"retrying":retried,"rejected":rejected}

    def select_audit_before_scoring(self, request: EvaluationRequest, review_context: dict, *, frozen_question_hash: str, kind: str = "random") -> dict:
        if not self.metadata_upload_enabled:
            raise PermissionError("Audit metadata upload has not been explicitly enabled")
        if request.workspaceId != self.workspace_id or kind not in ("random","mandatory"):
            raise ValueError("Pre-scoring audits require this workspace and random/mandatory selection")
        context=_review_context(review_context)
        bindings={"sourceContentHash":"sha256:"+hashlib.sha256(request.input.context.encode()).hexdigest(),
            "suggestedContentHash":"sha256:"+hashlib.sha256(request.input.text.encode()).hexdigest(),
            "frozenQuestionHash":_hash(frozen_question_hash)}
        key=request.input_commitment()
        with self.learning.transaction() as database:
            self._case_live(database,request.caseId)
            state=self._state(database)
            if key in state["results"] or key in database["evaluations"]:
                raise PermissionError("This case has already been scored; independent pre-scoring audit is unavailable")
        response=self._request("POST","/audits",json={"caseId":request.caseId,"modelBundleId":request.modelBundleId,
            "templateCommitment":request.template_commitment(),"inputCommitment":key,"agentId":self.agent_id,
            "agentVersionId":self.agent_version_id,"kind":kind,"aiExposed":False,"reviewContext":context,**bindings})
        if (response.get("kind") != kind or response.get("aiExposed") is not False or type(response.get("selected")) is not bool
            or response.get("blindingAssurance") != "connector_attested"):
            raise ValueError("Audit response does not establish the requested pre-scoring selection")
        _opaque(response.get("auditId"))
        probability=response.get("selectionProbabilityBps")
        if type(probability) is not int or not 1 <= probability <= 10000:
            raise ValueError("Audit selection probability is invalid")
        with self.learning.transaction() as database:
            self._case_live(database,request.caseId)
            self._state(database)["audits"][key]={"response":response,"case_id":request.caseId,
                "model_bundle_id":request.modelBundleId,"template_commitment":request.template_commitment(),
                "selected_before_scoring":True,"ai_exposed":False,"selected_at":time.time(),**bindings}
        return response

    def run_with_audit(self, request: EvaluationRequest, evaluate: Callable[[EvaluationRequest],EvaluationResult | dict],
                       review_context: dict, *, frozen_question_hash: str, allow_offline: bool = False) -> dict:
        audit=None
        try:
            audit=self.select_audit_before_scoring(request,review_context,frozen_question_hash=frozen_question_hash)
        except ConnectorUnavailable:
            if not allow_offline:
                raise
            with self.learning.transaction() as database:
                known = self._state(database)
                if known["mode"] != "shadow" or time.time()-known.get("synced_at",0) > 86400:
                    raise PermissionError("Offline connected evaluation requires a recent enabled workspace state") from None
        result=EvaluationResult.model_validate(evaluate(request))
        if (result.workspaceId,result.caseId,result.modelBundleId,result.inputCommitment,result.templateCommitment) != (
            request.workspaceId,request.caseId,request.modelBundleId,request.input_commitment(),request.template_commitment()):
            raise ValueError("Local evaluator returned a result for a different committed request")
        selected=bool(audit and audit["selected"])
        with self.learning.transaction() as database:
            self._case_live(database,request.caseId)
            state=self._state(database)
            state["results"][result.inputCommitment]=result.model_dump()
            if not audit:
                state["audits"][result.inputCommitment]={"selected_before_scoring":False,"ai_exposed":True,"offline":True}
            elif not selected:
                state["audits"][result.inputCommitment]["ai_exposed"]=True
        self.queue_result(result)
        return {"result":None if selected else result.model_dump(),"audit":audit,
                "awaitingIndependentHuman":selected,"blindingAssurance":"connector_attested" if selected else "none",
                "hostedHumanReviewReduction":False}

    def release_result(self, input_commitment: str) -> dict:
        """Explicit release marks exposure; import human labels before releasing."""
        with self.learning.transaction() as database:
            state=self._state(database)
            if input_commitment not in state["results"]:
                raise KeyError("Connector result not found")
            state["audits"].setdefault(input_commitment,{})["ai_exposed"]=True
            return deepcopy(state["results"][input_commitment])

    def fetch_and_import_labels(self, grant_id: str, *, question_id: str, template_commitment: str,
                                outcome_labels: dict[str,str]) -> dict:
        """Import overall human verdicts only through an explicit single-question mapping.

        This does not manufacture criterion labels from an overall verdict. A
        multi-question rubric requires separate authenticated human adjudication.
        """
        _opaque(grant_id); _opaque(question_id); _hash(template_commitment)
        if set(outcome_labels) != {"positive","negative"} or len(set(outcome_labels.values())) != 2:
            raise ValueError("Provide distinct explicit labels for positive and negative overall verdicts")
        self.sync_grants()
        response=self._request("GET","/labeled-data",params={"grantId":grant_id})
        body={k:v for k,v in response.items() if k != "exportDigest"}
        if response.get("exportDigest") != commitment(body,"rateloop.product-evaluator.v2"):
            raise ValueError("Human-label export commitment mismatch")
        if response.get("schemaVersion") != "rateloop.evaluator-labeled-data.v2" or response.get("workspaceId") != self.workspace_id or response.get("contentMode") != "commitments_only":
            raise ValueError("Unsupported human-label export")
        current=time.time()
        with self.learning.transaction() as database:
            state=deepcopy(self._state(database))
        if response.get("revocationWatermark") != state["watermark"]:
            raise PermissionError("Label export requires a fresh matching revocation watermark; synchronize again")
        if response.get("consent") is not None:
            remote=state.get("consents",{}).get(grant_id)
            permission=self._validate_consent(response["consent"],now=current,watermark=state["watermark"])
            lease=state.get("authorization_lease") or {}
            if (not remote or permission != remote["consent"] or permission["purpose"] != "private_learning"
                    or template_commitment not in permission["templateCommitments"] or "human_labels" not in permission["fields"]
                    or permission["revokedAt"] is not None or _timestamp(lease.get("expiresAt")) <= current
                    or (permission["expiresAt"] is not None and _timestamp(permission["expiresAt"]) <= current)):
                raise PermissionError("Label export lacks an active exact scoped human-label consent")
            permitted_bundles=permission["modelBundleIds"]
        else:
            remote=state["grants"].get(grant_id)
            permission=self._validate_grant(response.get("grant"),now=current,watermark=state["watermark"])
            if (not remote or permission != remote["grant"] or permission["templateCommitment"] != template_commitment
                or "human_labels" not in permission["fields"] or permission["revokedAt"] is not None or _timestamp(permission["expiresAt"]) <= current):
                raise PermissionError("Label export lacks an active exact scoped human-label grant")
            permitted_bundles=[permission["modelBundleId"]]
        if not isinstance(response.get("items"),list) or len(response["items"]) > 5000:
            raise ValueError("Unbounded label export")
        imported,rejected,duplicates=0,[],0
        for item in response["items"]:
            item_key=commitment(item,"rateloop.imported-human-label.v1")
            try:
                with self.learning.transaction() as database:
                    latest=self._state(database)
                    if item_key in latest["imports"]:
                        duplicates+=1
                        continue
                    row=deepcopy(database["evaluations"].get(item.get("inputCommitment")))
                    audit=deepcopy(latest["audits"].get(item.get("inputCommitment")))
                    local_result=deepcopy(latest["results"].get(item.get("inputCommitment")))
                if not row or row["workspace_id"] != self.workspace_id or not audit or not local_result:
                    raise ValueError("No matching local evaluation and pre-scoring audit")
                if (item.get("caseId"),item.get("modelBundleId"),item.get("templateCommitment")) != (
                    row["case_id"],row.get("model_bundle_id"),row["template_commitment"]):
                    raise ValueError("Human result does not bind the exact local case, model and template")
                if item.get("resultCommitment") != local_result["resultCommitment"]:
                    raise ValueError("Human export references a different automated result")
                if item.get("templateCommitment") != template_commitment or item.get("modelBundleId") not in permitted_bundles:
                    raise ValueError("Human result is outside the exported grant scope")
                if item.get("labelScope") != "overall_human_verdict" or item.get("criterionTrainingRequiresAdjudication") is not True:
                    raise ValueError("Unknown human-label semantics")
                questions=row["template"]["questions"]
                if len(questions)!=1 or questions[0]["id"] != question_id or not set(outcome_labels.values()) <= {x["id"] for x in questions[0]["labels"]}:
                    raise ValueError("Overall verdict mapping requires exactly one matching question")
                remote_audit=item.get("audit",{})
                causal=False
                if remote_audit.get("blindingAssurance") == "server_enforced":
                    frozen=_timestamp(remote_audit.get("reviewFrozenAt"))
                    released=remote_audit.get("resultsReleasedAt")
                    causal=(remote_audit.get("independent") is True and (released is None or frozen <= _timestamp(released))
                            and audit["selected_at"]-300 <= frozen <= current+300)
                    for name in ("sourceContentHash","suggestedContentHash","frozenQuestionHash"):
                        if _hash(item.get(name)) != audit.get(name):
                            raise ValueError("Human label changed the frozen review content")
                    if item.get("questionId") != question_id:
                        raise ValueError("Human label uses a different frozen question")
                if (not audit.get("selected_before_scoring") or (audit.get("ai_exposed") is not False and not causal)
                    or audit.get("response",{}).get("selected") is not True
                    or remote_audit.get("auditId") != audit["response"]["auditId"]
                    or (remote_audit.get("aiExposed") is not False and not causal) or remote_audit.get("kind") not in ("random","mandatory")
                    or remote_audit.get("kind") != audit["response"]["kind"]
                    or remote_audit.get("selectionProbabilityBps") != audit["response"]["selectionProbabilityBps"]
                    or (remote_audit.get("blindingAssurance") == "server_enforced" and not causal)):
                    raise ValueError("Human label lacks independently blind connector provenance")
                if item.get("humanOutcome") not in outcome_labels or type(item.get("responseCount")) is not int or item["responseCount"] < 1:
                    raise ValueError("Inconclusive or missing human verdict")
                _hash(item.get("humanResultCommitment"))
                if not audit["selected_at"]-300 <= _timestamp(item.get("observedAt")) <= current+300:
                    raise ValueError("Human verdict time is outside the audit observation window")
                label=outcome_labels[item["humanOutcome"]]
                feedback=self.learning.add_feedback(workspace_id=self.workspace_id,evaluation_id=row["evaluation_id"],
                    input_commitment=row["input_commitment"],template_commitment=row["template_commitment"],
                    annotator_id="rateloop-consensus:"+item["humanResultCommitment"][7:47],labels={question_id:label},
                    exposed_to_ai=False,independent_human=True,feedback_id="import_"+item_key[7:])
                if feedback["quarantine_reasons"]:
                    raise ValueError("Imported verdict was quarantined: "+",".join(feedback["quarantine_reasons"]))
                with self.learning.transaction() as database:
                    self._state(database)["imports"][item_key]={"feedback_id":feedback["id"],"at":current}
                imported+=1
            except (ValueError,PermissionError,KeyError,TypeError) as error:
                rejected.append({"itemCommitment":item_key,"reason":str(error)})
        return {"imported":imported,"duplicates":duplicates,"rejected":rejected,"truncated":response.get("truncated") is True,
                "labelProvenance":"Authenticated RateLoop overall human consensus, mapped explicitly to one question."}
