"""Authenticated single-worker local service; a real backend is injected at startup."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import hmac
import threading
import time
from typing import Protocol

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import Field

from .calibration import apply_temperature
from .learning import LearningStore
from .protocol import EvaluationRequest, EvaluationResult, WireModel, Identifier, Digest, commitment, make_result, utc_now
from .storage import RuntimeStore


class Backend(Protocol):
    def predict(self, text: str, questions: list[dict]) -> dict[str,dict[str,float]]: ...
    def count_tokens(self, text: str, questions: list[dict]) -> int: ...


class Feedback(WireModel):
    workspaceId: Identifier
    evaluationId: Identifier
    inputCommitment: Digest
    templateCommitment: Digest
    annotatorId: Identifier
    labels: dict[Identifier, Identifier]
    exposedToAi: bool
    independentHuman: bool


@dataclass(frozen=True)
class Principal:
    workspace_id: str
    roles: frozenset[str]
    annotator_id: str | None = None


class BodyLimit:
    def __init__(self, app, max_bytes=350_000): self.app, self.max_bytes = app, max_bytes
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http": return await self.app(scope, receive, send)
        chunks = []; size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect": return
            size += len(message.get("body", b""))
            if size > self.max_bytes:
                return await JSONResponse({"code":"body_too_large"},status_code=413)(scope, receive, send)
            chunks.append(message)
            if not message.get("more_body", False): break
        async def replay(): return chunks.pop(0) if chunks else {"type":"http.disconnect"}
        await self.app(scope, replay, send)


def create_app(*, backend: Backend, bundle: dict, learning: LearningStore, runtime: RuntimeStore,
               tokens: dict[str,Principal], validate_bundle=None, allow_training_retention=None) -> FastAPI:
    """tokens maps SHA256 token hashes to workspace roles; raw tokens are never persisted.

    validate_bundle rechecks registry signature/lineage and returns current mode.
    Serving without it is shadow-only, so a loaded checkpoint cannot auto-enable review reduction.
    """
    if not tokens or any(len(key) != 64 for key in tokens): raise ValueError("Scoped token hashes are required")
    app = FastAPI(title="RateLoop Evaluator", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(BodyLimit)
    worker = threading.Lock()

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, error):
        # Pydantic's default response echoes rejected customer input.
        return JSONResponse({"code":"invalid_request"}, status_code=422)

    def principal(request: Request) -> Principal:
        if request.headers.get("origin") is not None:
            raise HTTPException(403, detail="Browser origins are not enabled")
        authorization = request.headers.get("authorization", "")
        if not authorization.startswith("Bearer ") or len(authorization) > 512:
            raise HTTPException(401, detail="Scoped bearer token required")
        digest = hashlib.sha256(authorization[7:].encode()).hexdigest()
        for expected, identity in tokens.items():
            if hmac.compare_digest(expected,digest): return identity
        raise HTTPException(401, detail="Invalid token")

    @app.get("/v1/capabilities")
    def capabilities(identity: Principal = Depends(principal)):
        if "evaluate" not in identity.roles: raise HTTPException(403,detail="Evaluation scope required")
        return {"schemaVersion":"rateloop.evaluator.capabilities.v1", "modelBundleId":bundle["id"],
                "languages":bundle["languages"],"templateCommitments":bundle["template_commitments"],
                "maxTokens":bundle.get("max_tokens",512),"training":"local_privileged_cli",
                "defaultMode":"shadow", "rawContentUploads":False}

    @app.post("/v1/evaluate")
    def evaluate(request: EvaluationRequest, identity: Principal = Depends(principal)):
        if "evaluate" not in identity.roles or request.workspaceId != identity.workspace_id:
            raise HTTPException(403, detail="Workspace or evaluation scope mismatch")
        start = time.monotonic()
        fields = ["input."+name for name,value in request.input.model_dump().items() if value]
        def require_right(right):
            return learning.check_right(workspace_id=request.workspaceId,right=right,case_id=request.caseId,
                                        template_id=request.template.id,fields=fields,model_bundle_id=request.modelBundleId,
                                        template_commitment=request.template_commitment())
        try: require_right("ai_use")
        except PermissionError: raise HTTPException(403,detail="AI-use grant required") from None
        if request.modelBundleId != bundle["id"]: raise HTTPException(409,detail="Model bundle mismatch")
        template_digest = request.template_commitment(); input_digest = request.input_commitment()
        if not worker.acquire(blocking=False): raise HTTPException(429,detail="Worker busy; retry the same idempotency key")
        try:
            deployment = validate_bundle(request) if validate_bundle else {"mode":"shadow"}
            mode = deployment["mode"]
            policy_digest = commitment(deployment,"rateloop.serving-policy.v1")
            cached = runtime.get(request.workspaceId,request.idempotencyKey,input_digest)
            if cached is not None:
                if cached["policyCommitment"] != policy_digest:
                    raise HTTPException(409,detail="Deployment policy changed; use a new evaluation key")
                return EvaluationResult.model_validate(cached["result"]).model_dump()
            reason = None; criteria = []; outcome = "uncertain"
            if request.template.language not in bundle["languages"]: reason = "unsupported_language"
            elif template_digest not in bundle["template_commitments"]: reason = "unsupported_template"
            else:
                questions = [q.model_dump() for q in request.template.questions]
                text = request.input.render()
                token_count = backend.count_tokens(text,questions)
                if token_count > min(request.template.maxTokens,bundle.get("max_tokens",512)): reason = "input_too_long"
                elif (time.monotonic()-start)*1000 >= request.deadlineMs: reason = "deadline_exceeded"
                else:
                    scores = backend.predict(text,questions)
                    if set(scores) != {q.id for q in request.template.questions}: raise ValueError("Backend question mismatch")
                    calibrations = {c["question_id"]:c for c in bundle.get("calibrations",[]) if c["template_commitment"] == template_digest and c["language"] == request.template.language}
                    for question in request.template.questions:
                        raw = scores[question.id]
                        if set(raw) != {label.id for label in question.labels}: raise ValueError("Backend label mismatch")
                        calibration = calibrations.get(question.id); probabilities = None
                        if calibration:
                            probabilities = apply_temperature(raw,calibration,model_bundle_id=bundle["id"],template_commitment=template_digest,question_id=question.id,language=request.template.language)
                        chosen = max(probabilities or raw, key=(probabilities or raw).get)
                        criteria.append({"questionId":question.id,"label":chosen,"rawScores":raw,
                                         "probabilities":probabilities,"calibrationId":calibration["id"] if calibration else None})
                    reason = "uncalibrated" if any(c["probabilities"] is None for c in criteria) else "shadow_only"
                    if mode == "selective" and reason != "uncalibrated":
                        threshold = deployment["gate"]["threshold"]
                        if not 0 < threshold <= 1: raise ValueError("Invalid decision threshold")
                        if all(q.passLabels for q in request.template.questions) and all(c["probabilities"][c["label"]] >= threshold for c in criteria):
                            outcome = "pass" if all(c["label"] in q.passLabels for c,q in zip(criteria,request.template.questions)) else "fail"
                            reason = None
                        else: reason = "low_confidence"
            duration = int((time.monotonic()-start)*1000)
            if duration >= request.deadlineMs:
                outcome,reason = "uncertain","deadline_exceeded"
            # Revocation during inference must prevent result release and retention.
            require_right("ai_use")
            if validate_bundle and commitment(validate_bundle(request),"rateloop.serving-policy.v1") != policy_digest:
                raise HTTPException(409,detail="Deployment policy changed during evaluation")
            result = make_result(workspaceId=request.workspaceId,caseId=request.caseId,modelBundleId=bundle["id"],
                                 inputCommitment=input_digest,templateCommitment=template_digest,outcome=outcome,
                                 abstainReason=reason,criteria=criteria,durationMs=duration,observedAt=utc_now())
            retained = None
            try:
                require_right("private_training")
                if allow_training_retention is None or allow_training_retention(request):
                    retained = request.input.model_dump()
            except PermissionError: pass
            learning.record_evaluation(evaluation_id=input_digest,workspace_id=request.workspaceId,case_id=request.caseId,
                input_commitment=input_digest,template_commitment=template_digest,template=request.template.model_dump(),
                input_payload=retained,fields=fields,group_id=request.sourceGroupId,model_bundle_id=request.modelBundleId)
            saved = runtime.put(request.workspaceId,request.idempotencyKey,input_digest,{"result":result.model_dump(),"policyCommitment":policy_digest})
            return saved["result"]
        except HTTPException:
            raise
        except PermissionError:
            raise HTTPException(403,detail="Grant or model authorization expired") from None
        except KeyError:
            raise HTTPException(409,detail="Model bundle unavailable") from None
        except ValueError as error:
            if "Idempotency" in str(error): raise HTTPException(409,detail="Idempotency conflict") from None
            raise HTTPException(503,detail="Evaluator validation failed") from None
        except Exception:
            raise HTTPException(503,detail="Evaluation unavailable; human review required") from None
        finally: worker.release()

    @app.post("/v1/feedback")
    def feedback(request: Feedback, identity: Principal = Depends(principal)):
        if "feedback" not in identity.roles or identity.workspace_id != request.workspaceId or not identity.annotator_id or identity.annotator_id != request.annotatorId:
            raise HTTPException(403,detail="Workspace or feedback scope mismatch")
        try:
            row = learning.add_feedback(workspace_id=request.workspaceId,evaluation_id=request.evaluationId,
                input_commitment=request.inputCommitment,template_commitment=request.templateCommitment,
                annotator_id=request.annotatorId,labels=request.labels,exposed_to_ai=request.exposedToAi,
                independent_human=request.independentHuman)
            return {"feedbackId":row["id"],"quarantineReasons":row["quarantine_reasons"]}
        except PermissionError: raise HTTPException(403,detail="Private-training grant required") from None
        except KeyError: raise HTTPException(404,detail="Evaluation not found") from None
        except ValueError: raise HTTPException(422,detail="Invalid feedback") from None

    # The outbound worker calls the same authenticated evaluation core in-process;
    # it does not expose a second listener or duplicate inference policy.
    app.state.evaluate = evaluate
    return app
