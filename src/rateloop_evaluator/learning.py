"""Encrypted local feedback, purpose-scoped grants and immutable training snapshots.

The supplied key belongs to the operator and must be stored separately from public
code and backups. Revocation prevents future use and retires descendants; it does
not promise to remove information from copies of already distributed weights.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import threading
import time
from typing import Any, Iterator
import uuid

from cryptography.fernet import Fernet

RIGHTS = frozenset({"ai_use", "private_training", "shared_contribution", "public_weight_distribution"})
COLLECTIONS = ("grants", "evaluations", "feedback", "snapshots", "lineage", "bundles", "deployments")


def provision_key(path: str | Path) -> None:
    """Create a customer-owned encryption key exclusively, never overwrite it."""
    fd = os.open(Path(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(Fernet.generate_key())
        handle.flush()
        os.fsync(handle.fileno())


def read_secret(path: str | Path) -> bytes:
    path = Path(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
        raise ValueError("Key must be a regular non-symlink file accessible only to its owner")
    with path.open("rb") as handle:
        return handle.read().strip()


def _json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value)).hexdigest()


class LearningStore:
    """Single-host encrypted store with atomic writes and process-safe locking.

    There are no network calls. All identifiers, feedback, raw content, snapshots,
    model lineage and registry state are encrypted together at rest. Callers must
    not log input arguments; local admins controlling the running process can read
    decrypted memory. Do not place this store on unsupported network filesystems.
    """
    def __init__(self, root: str | Path, key_file: str | Path):
        self.root = Path(root)
        if self.root.is_symlink():
            raise ValueError("Store directory cannot be a symlink")
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        self._fernet = Fernet(read_secret(key_file))
        self._mutex = threading.RLock()
        self._path = self.root / "state.enc"
        self._lock_path = self.root / "state.lock"
        if self._path.is_symlink() or self._lock_path.is_symlink():
            raise ValueError("Store files cannot be symlinks")
        with self.transaction():
            pass

    @contextmanager
    def transaction(self) -> Iterator[dict[str, Any]]:
        """Internal transaction API also used by the signed bundle registry."""
        with self._mutex:
            fd = os.open(self._lock_path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
            with os.fdopen(fd, "a+b") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if self._path.exists():
                    if self._path.is_symlink():
                        raise ValueError("Encrypted state cannot be a symlink")
                    state = json.loads(self._fernet.decrypt(self._path.read_bytes()))
                else:
                    state = {"schema_version": 1, **{k: {} for k in COLLECTIONS}}
                original = _json(state)
                yield state
                updated = _json(state)
                if updated == original and self._path.exists():
                    return
                payload = self._fernet.encrypt(updated)
                out_fd, name = tempfile.mkstemp(prefix=".state-", dir=self.root)
                try:
                    with os.fdopen(out_fd, "wb") as target:
                        target.write(payload)
                        target.flush()
                        os.fsync(target.fileno())
                    os.replace(name, self._path)
                finally:
                    if os.path.exists(name):
                        os.unlink(name)

    def add_grant(self, *, workspace_id: str, rights: list[str], expires_at: float,
                  case_ids: list[str] | None = None, template_ids: list[str] | None = None,
                  fields: list[str] | None = None, grant_id: str | None = None,
                  model_bundle_ids: list[str] | None = None, template_commitments: list[str] | None = None,
                  authorization_until: float | None = None,
                  evidence: str, now: float | None = None) -> dict[str, Any]:
        """Record an explicit grant; None scope means all, [] means no items.

        Rights are independent: AI permission does not authorize retention/training,
        private training does not authorize shared data or public weight release.
        `evidence` is a locally retained reference to the operator's authorization.
        """
        current = time.time() if now is None else now
        if not workspace_id or not evidence or not rights or not set(rights) <= RIGHTS:
            raise ValueError("An evidenced, workspace-bound grant with known rights is required")
        if not isinstance(expires_at, (float, int)) or not current < expires_at < float("inf"):
            raise ValueError("Grant expiration must be a finite future timestamp")
        if authorization_until is not None and not current < authorization_until <= min(expires_at,current+900):
            raise ValueError("Worker authorization must expire within 15 minutes")
        for scope in (case_ids, template_ids, fields, model_bundle_ids, template_commitments):
            if scope is not None and (not isinstance(scope, list) or any(not isinstance(x, str) or not x for x in scope)):
                raise ValueError("Scope entries must be nonempty strings")
        grant = {"id": grant_id or "grant_"+uuid.uuid4().hex, "workspace_id": workspace_id,
                 "rights": sorted(set(rights)), "expires_at": expires_at, "case_ids": case_ids,
                 "template_ids": template_ids, "fields": fields, "evidence": evidence,
                 "model_bundle_ids": model_bundle_ids, "template_commitments": template_commitments,
                 "created_at": current, "revoked_at": None, "authorization_until": authorization_until}
        with self.transaction() as state:
            if grant["id"] in state["grants"]:
                raise ValueError("Grant IDs are immutable")
            state["grants"][grant["id"]] = grant
        return deepcopy(grant)

    def renew_authorization(self, grant_id: str, workspace_id: str, expires_at: float, *, now: float | None = None) -> None:
        """Renew execution only; an expired or revoked durable consent never revives."""
        current = time.time() if now is None else now
        with self.transaction() as state:
            grant = state["grants"].get(grant_id)
            if (not grant or grant["workspace_id"] != workspace_id or grant["revoked_at"] is not None
                    or grant["expires_at"] <= current or grant.get("authorization_until") is None):
                raise PermissionError("Durable consent is unavailable for renewal")
            if not current < expires_at <= min(grant["expires_at"],current+900):
                raise ValueError("Worker authorization must expire within 15 minutes")
            grant["authorization_until"] = expires_at

    @staticmethod
    def _matching_grants(state: dict, *, workspace_id: str, right: str, case_id: str,
                         template_id: str, fields: list[str], now: float, model_bundle_id: str | None = None,
                         template_commitment: str | None = None) -> list[str]:
        if right not in RIGHTS or not workspace_id or not case_id or not template_id or not fields:
            raise PermissionError("A known right and complete evaluation scope are required")
        if _digest([workspace_id, case_id]) in state.get("deleted_cases", {}):
            raise PermissionError("This case was deleted; submit new work under a new case ID")
        matches = []
        for grant_id, grant in state["grants"].items():
            if grant["workspace_id"] != workspace_id or right not in grant["rights"] or grant["revoked_at"] is not None or grant["expires_at"] <= now:
                continue
            if grant.get("authorization_until") is not None and grant["authorization_until"] <= now:
                continue
            if grant["case_ids"] is not None and case_id not in grant["case_ids"]:
                continue
            if grant["template_ids"] is not None and template_id not in grant["template_ids"]:
                continue
            if grant["fields"] is not None and not set(fields) <= set(grant["fields"]):
                continue
            if grant.get("model_bundle_ids") is not None and model_bundle_id not in grant["model_bundle_ids"]:
                continue
            if grant.get("template_commitments") is not None and template_commitment not in grant["template_commitments"]:
                continue
            matches.append(grant_id)
        if not matches:
            raise PermissionError(f"No active {right} grant covers this scope")
        return sorted(matches)

    def check_right(self, *, workspace_id: str, right: str, case_id: str,
                    template_id: str, fields: list[str], now: float | None = None,
                    model_bundle_id: str | None = None, template_commitment: str | None = None) -> list[str]:
        with self.transaction() as state:
            return self._matching_grants(state, workspace_id=workspace_id, right=right, case_id=case_id,
                                         template_id=template_id, fields=fields, now=time.time() if now is None else now,
                                         model_bundle_id=model_bundle_id, template_commitment=template_commitment)

    def revoke_grant(self, grant_id: str, workspace_id: str, *, now: float | None = None) -> dict:
        current = time.time() if now is None else now
        with self.transaction() as state:
            grant = state["grants"].get(grant_id)
            if not grant or grant["workspace_id"] != workspace_id:
                raise KeyError("Grant not found")
            grant["revoked_at"] = current
            affected_snapshots, retired_bundles = [], []
            for sid, snapshot in state["snapshots"].items():
                if grant_id in snapshot["grant_ids"]:
                    snapshot["invalidated_at"] = current
                    snapshot["invalidation_reason"] = "source_grant_revoked"
                    affected_snapshots.append(sid)
            for bundle_id, lineage in state["lineage"].items():
                if lineage["snapshot_id"] in affected_snapshots:
                    lineage["retired_at"] = current
                    lineage["retirement_reason"] = "source_grant_revoked"
                    retired_bundles.append(bundle_id)
            return {"invalidated_snapshots": affected_snapshots, "retired_bundles": retired_bundles,
                    "unlearning": "Retirement prevents future managed use; distributed copies are not erased."}

    def record_evaluation(self, *, evaluation_id: str, workspace_id: str, case_id: str,
                          input_commitment: str, template_commitment: str, template: dict,
                          input_payload: dict | None = None, group_id: str | None = None,
                          fields: list[str] | None = None, now: float | None = None,
                          model_bundle_id: str | None = None) -> dict:
        current = time.time() if now is None else now
        fields = fields or ["input.text", "input.context", "input.evidence"]
        if not all((evaluation_id, workspace_id, case_id, input_commitment, template_commitment)):
            raise ValueError("Evaluation identity and commitments are required")
        if not template.get("id") or not template.get("version") or not template.get("questions"):
            raise ValueError("A versioned question template is required")
        if not all(q.get("id") and len(q.get("labels", [])) >= 2 for q in template["questions"]):
            raise ValueError("Template questions must have IDs and answer labels")
        if input_payload is not None and not {"input."+k for k,v in input_payload.items() if v} <= set(fields):
            raise ValueError("Every retained input field must be covered by the requested grant scope")
        with self.transaction() as state:
            self._matching_grants(state, workspace_id=workspace_id, right="ai_use", case_id=case_id,
                                  template_id=template["id"], fields=fields, now=current,
                                  model_bundle_id=model_bundle_id, template_commitment=template_commitment)
            if input_payload is not None:
                self._matching_grants(state, workspace_id=workspace_id, right="private_training", case_id=case_id,
                                      template_id=template["id"], fields=fields, now=current,
                                  model_bundle_id=model_bundle_id, template_commitment=template_commitment)
            row = {"evaluation_id": evaluation_id, "workspace_id": workspace_id, "case_id": case_id,
                   "input_commitment": input_commitment, "template_commitment": template_commitment,
                   "template": deepcopy(template), "input": deepcopy(input_payload), "group_id": group_id or case_id,
                   "model_bundle_id": model_bundle_id,
                   "fields": list(fields), "created_at": current, "retained_at": current if input_payload is not None else None}
            existing = state["evaluations"].get(evaluation_id)
            if existing:
                immutable = {"evaluation_id", "workspace_id", "case_id", "input_commitment", "template_commitment",
                             "template", "group_id", "model_bundle_id", "fields"}
                if any(existing.get(k) != row.get(k) for k in immutable):
                    raise ValueError("Evaluation ID is already bound to different content")
                if input_payload is not None:
                    if existing["input"] is not None and existing["input"] != input_payload:
                        raise ValueError("Evaluation ID is already bound to different retained content")
                    if existing["input"] is None:
                        # A later explicit grant authorizes retention from this
                        # new submission, never a retroactive hidden capture.
                        existing["input"] = deepcopy(input_payload)
                        existing["retained_at"] = current
                response = deepcopy(existing)
                if input_payload is None:
                    response["input"] = None
                return response
            state["evaluations"][evaluation_id] = row
        return deepcopy(row)

    def add_feedback(self, *, workspace_id: str, evaluation_id: str, input_commitment: str,
                     template_commitment: str, annotator_id: str, labels: dict[str, str],
                     exposed_to_ai: bool, independent_human: bool, feedback_id: str | None = None,
                     now: float | None = None) -> dict:
        """Preserve all submitted labels; quarantine unsafe or contradictory data.

        Human provenance is an authenticated caller attestation, not inferred from
        text. A deployment must authenticate reviewers before using this method.
        """
        current = time.time() if now is None else now
        if not annotator_id or type(exposed_to_ai) is not bool or type(independent_human) is not bool:
            raise ValueError("Reviewer identity and explicit provenance flags are required")
        with self.transaction() as state:
            row = state["evaluations"].get(evaluation_id)
            if not row or row["workspace_id"] != workspace_id:
                raise KeyError("Evaluation not found")
            self._matching_grants(state, workspace_id=workspace_id, right="private_training", case_id=row["case_id"],
                                  template_id=row["template"]["id"], fields=[*row["fields"], "human_labels"], now=current,
                                  model_bundle_id=row.get("model_bundle_id"), template_commitment=row["template_commitment"])
            reasons = []
            if input_commitment != row["input_commitment"] or template_commitment != row["template_commitment"]:
                reasons.append("commitment_mismatch")
            expected = {q["id"]: {label["id"] for label in q["labels"]} for q in row["template"]["questions"]}
            if not isinstance(labels, dict) or set(labels) != set(expected) or any(v not in expected[k] for k,v in labels.items() if k in expected):
                reasons.append("invalid_labels")
            if exposed_to_ai or not independent_human:
                reasons.append("not_independent_human")
            if any(f["evaluation_id"] == evaluation_id and f["annotator_id"] == annotator_id for f in state["feedback"].values()):
                reasons.append("duplicate_reviewer")
            feedback = {"id": feedback_id or "feedback_"+uuid.uuid4().hex, "workspace_id": workspace_id,
                        "evaluation_id": evaluation_id, "annotator_id": annotator_id, "labels": deepcopy(labels),
                        "input_commitment": input_commitment, "template_commitment": template_commitment,
                        "exposed_to_ai": exposed_to_ai, "independent_human": independent_human,
                        "quarantine_reasons": reasons, "created_at": current}
            if feedback["id"] in state["feedback"]:
                raise ValueError("Feedback IDs are immutable")
            if not reasons:
                valid = [f for f in state["feedback"].values() if f["evaluation_id"] == evaluation_id and
                         not set(f["quarantine_reasons"]) - {"human_disagreement"}]
                if any(f["labels"] != labels for f in valid):
                    feedback["quarantine_reasons"].append("human_disagreement")
                    for f in valid:
                        if "human_disagreement" not in f["quarantine_reasons"]:
                            f["quarantine_reasons"].append("human_disagreement")
            state["feedback"][feedback["id"]] = feedback
            # Earlier snapshots cannot silently keep superseded consensus labels.
            if "human_disagreement" in feedback["quarantine_reasons"]:
                self._invalidate_evaluation(state, evaluation_id, current, "human_disagreement")
            return deepcopy(feedback)

    @staticmethod
    def _invalidate_evaluation(state: dict, evaluation_id: str, now: float, reason: str) -> None:
        affected = []
        for sid, snapshot in state["snapshots"].items():
            if any(e["evaluation_id"] == evaluation_id for part in ("train", "calibration", "test") for e in snapshot[part]):
                snapshot["invalidated_at"] = now
                snapshot["invalidation_reason"] = reason
                affected.append(sid)
        for lineage in state["lineage"].values():
            if lineage["snapshot_id"] in affected:
                lineage["retired_at"] = now
                lineage["retirement_reason"] = reason

    def create_snapshot(self, workspace_id: str, template_id: str, template_version: int | str, *,
                        train_fraction: float = .7, calibration_fraction: float = .15, seed: str = "rateloop-v1",
                        minimum_groups: int = 3, purpose: str = "private_training", now: float | None = None,
                        template_commitment: str | None = None) -> dict:
        current = time.time() if now is None else now
        if purpose not in ("private_training", "shared_contribution", "public_weight_distribution"):
            raise ValueError("Invalid snapshot purpose")
        if not 0 < train_fraction < 1 or not 0 < calibration_fraction < 1 or train_fraction+calibration_fraction >= 1:
            raise ValueError("All three partitions require a positive fraction")
        with self.transaction() as state:
            examples, grant_ids = [], set()
            for row in state["evaluations"].values():
                if row["workspace_id"] != workspace_id or row["template"]["id"] != template_id or row["template"]["version"] != template_version or row["input"] is None:
                    continue
                if template_commitment is not None and row["template_commitment"] != template_commitment:
                    continue
                labels = [f for f in state["feedback"].values() if f["evaluation_id"] == row["evaluation_id"] and not f["quarantine_reasons"]]
                if not labels:
                    continue
                try:
                    source_grants = set(self._matching_grants(state, workspace_id=workspace_id, right="private_training",
                                    case_id=row["case_id"], template_id=template_id, fields=[*row["fields"], "human_labels"], now=current,
                                    model_bundle_id=row.get("model_bundle_id"), template_commitment=row["template_commitment"]))
                    extra_rights = set() if purpose == "private_training" else {purpose}
                    for right in extra_rights:
                        source_grants.update(self._matching_grants(state, workspace_id=workspace_id, right=right,
                                      case_id=row["case_id"], template_id=template_id, fields=[*row["fields"], "human_labels"], now=current,
                                    model_bundle_id=row.get("model_bundle_id"), template_commitment=row["template_commitment"]))
                except PermissionError:
                    continue
                example = deepcopy(row)
                example.update({"labels": deepcopy(labels[0]["labels"]), "human_labels": deepcopy(labels), "grant_ids": sorted(source_grants)})
                grant_ids.update(source_grants)
                examples.append(example)
            if len({r["template_commitment"] for r in examples}) > 1:
                raise ValueError("Template version maps to conflicting committed definitions")
            # Union cases, caller-supplied source groups and exact duplicate inputs,
            # preventing retries/revisions/duplicate text from crossing partitions.
            parent = list(range(len(examples)))
            def find(i: int) -> int:
                while parent[i] != i:
                    parent[i] = parent[parent[i]]
                    i = parent[i]
                return i
            seen: dict[str, int] = {}
            for i, row in enumerate(examples):
                for key in ("case:"+row["case_id"], "group:"+row["group_id"], "input:"+_digest(row["input"])):
                    if key in seen:
                        parent[find(i)] = find(seen[key])
                    seen[key] = i
            groups: dict[int, list[dict]] = {}
            for i, row in enumerate(examples):
                groups.setdefault(find(i), []).append(row)
            grouped = []
            for rows in groups.values():
                group = "group_"+_digest(sorted(r["evaluation_id"] for r in rows))
                for row in rows:
                    row["group_id"] = group
                grouped.append((hashlib.sha256((seed+group).encode()).hexdigest(), rows))
            grouped.sort(key=lambda x: x[0])
            if len(grouped) < max(3, minimum_groups):
                raise ValueError("Insufficient independent eligible groups for train/calibration/test")
            n_train = min(len(grouped)-2, max(1, int(len(grouped)*train_fraction)))
            n_cal = min(len(grouped)-n_train-1, max(1, int(len(grouped)*calibration_fraction)))
            partitions = {"train": grouped[:n_train], "calibration": grouped[n_train:n_train+n_cal], "test": grouped[n_train+n_cal:]}
            snapshot = {"id": "snapshot_"+uuid.uuid4().hex, "workspace_id": workspace_id,
                        "template_id": template_id, "template_version": template_version, "purpose": purpose,
                        "created_at": current, "seed": seed, "grant_ids": sorted(grant_ids), "invalidated_at": None,
                        "group_count": len(grouped), **{k: [row for _, rows in part for row in rows] for k,part in partitions.items()}}
            snapshot["content_digest"] = _digest({k: snapshot[k] for k in ("workspace_id", "template_id", "template_version", "purpose", "train", "calibration", "test")})
            state["snapshots"][snapshot["id"]] = snapshot
            return deepcopy(snapshot)

    @staticmethod
    def _validate_snapshot(state: dict, snapshot: dict, workspace_id: str, now: float) -> None:
        if snapshot["workspace_id"] != workspace_id:
            raise KeyError("Snapshot not found")
        if snapshot["invalidated_at"] is not None:
            raise PermissionError("Snapshot has been invalidated")
        digest = _digest({k: snapshot[k] for k in ("workspace_id", "template_id", "template_version", "purpose", "train", "calibration", "test")})
        if digest != snapshot["content_digest"]:
            raise ValueError("Snapshot digest mismatch")
        # Preserve the original authorization lineage. A new grant never silently
        # reauthorizes an old snapshot whose original permission has expired.
        for grant_id in snapshot["grant_ids"]:
            grant = state["grants"].get(grant_id)
            if not grant or grant["workspace_id"] != workspace_id or grant["revoked_at"] is not None or grant["expires_at"] <= now:
                raise PermissionError("Snapshot source grant is expired or revoked")
            if grant.get("authorization_until") is not None and grant["authorization_until"] <= now:
                raise PermissionError("Snapshot worker authorization requires renewal")
        for part in ("train", "calibration", "test"):
            for row in snapshot[part]:
                required_rights = {"private_training", snapshot["purpose"]}
                for right in required_rights:
                    LearningStore._matching_grants(state, workspace_id=workspace_id, right=right, case_id=row["case_id"],
                                                   template_id=snapshot["template_id"], fields=[*row["fields"], "human_labels"], now=now,
                                                   model_bundle_id=row.get("model_bundle_id"), template_commitment=row["template_commitment"])

    def load_snapshot(self, snapshot_id: str, workspace_id: str, *, now: float | None = None) -> dict:
        with self.transaction() as state:
            snapshot = state["snapshots"].get(snapshot_id)
            if not snapshot:
                raise KeyError("Snapshot not found")
            self._validate_snapshot(state, snapshot, workspace_id, time.time() if now is None else now)
            return deepcopy(snapshot)

    def register_model_lineage(self, bundle_id: str, snapshot_id: str, workspace_id: str, *, now: float | None = None) -> None:
        current = time.time() if now is None else now
        with self.transaction() as state:
            snapshot = state["snapshots"].get(snapshot_id)
            if not snapshot:
                raise KeyError("Snapshot not found")
            self._validate_snapshot(state, snapshot, workspace_id, current)
            if bundle_id in state["lineage"]:
                raise ValueError("Bundle lineage is immutable")
            state["lineage"][bundle_id] = {"workspace_id": workspace_id, "snapshot_id": snapshot_id,
                                           "registered_at": current, "retired_at": None}

    @staticmethod
    def _validate_lineage(state: dict, bundle_id: str, workspace_id: str, now: float) -> None:
        lineage = state["lineage"].get(bundle_id)
        if not lineage or lineage["workspace_id"] != workspace_id:
            raise KeyError("Model lineage not found")
        if lineage["retired_at"] is not None:
            raise PermissionError("Model is retired")
        LearningStore._validate_snapshot(state, state["snapshots"][lineage["snapshot_id"]], workspace_id, now)

    def assert_model_usable(self, bundle_id: str, workspace_id: str, *, now: float | None = None) -> None:
        with self.transaction() as state:
            self._validate_lineage(state, bundle_id, workspace_id, time.time() if now is None else now)

    def delete_case(self, workspace_id: str, case_id: str, *, now: float | None = None) -> dict:
        """Delete local raw content/labels and invalidate dependent managed models.

        Backups and externally distributed artifacts need separate operator handling.
        """
        current = time.time() if now is None else now
        with self.transaction() as state:
            state.setdefault("deleted_cases", {})[_digest([workspace_id, case_id])] = current
            ids = [k for k,v in state["evaluations"].items() if v["workspace_id"] == workspace_id and v["case_id"] == case_id]
            commitments = {state["evaluations"][key]["input_commitment"] for key in ids}
            feedback_ids = {k for k,v in state["feedback"].items() if v["evaluation_id"] in ids}
            for connector in state.get("connectors", {}).values():
                results = connector.get("results", {})
                removed = {k for k,v in results.items() if v.get("workspaceId") == workspace_id and v.get("caseId") == case_id}
                matching_keys = commitments | removed
                connector["results"] = {k:v for k,v in results.items() if k not in matching_keys}
                connector["audits"] = {k:v for k,v in connector.get("audits", {}).items()
                    if k not in matching_keys and not (connector.get("workspace_id") == workspace_id and v.get("case_id") == case_id)}
                connector["imports"] = {k:v for k,v in connector.get("imports", {}).items() if v.get("feedback_id") not in feedback_ids}
                for collection in ("worker_jobs","receipt_jobs","collections"):
                    connector[collection]={k:v for k,v in connector.get(collection,{}).items() if v.get("caseId")!=case_id}
            for evaluation_id in ids:
                self._invalidate_evaluation(state, evaluation_id, current, "source_deleted")
                del state["evaluations"][evaluation_id]
            state["feedback"] = {k:v for k,v in state["feedback"].items() if v["evaluation_id"] not in ids}
            for snapshot in state["snapshots"].values():
                for part in ("train", "calibration", "test"):
                    snapshot[part] = [r for r in snapshot[part] if r["evaluation_id"] not in ids]
            return {"deleted_evaluations": len(ids), "backups_and_distributed_weights": "Require separate operator handling; no instant unlearning is claimed."}
