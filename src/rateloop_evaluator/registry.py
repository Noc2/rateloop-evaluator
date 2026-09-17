"""Locally signed model bundles, measured promotion gates and safe rollback.

Signatures establish provenance under an operator-pinned key, not model quality.
Only independently calibrated and tested, non-synthetic bundles can be promoted
for selective automation. A base checkpoint can always be evaluated in shadow.
"""
from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath
import time
from typing import Any

import rfc8785

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .calibration import apply_temperature, false_approval_upper_bound, validate_calibration
from .learning import LearningStore, _json, read_secret


def provision_signing_key(path: str | Path) -> None:
    """Create an Ed25519 signing seed separately from the store encryption key."""
    key = Ed25519PrivateKey.generate().private_bytes(serialization.Encoding.Raw,
                                                    serialization.PrivateFormat.Raw,
                                                    serialization.NoEncryption())
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as handle:
        # Text encoding avoids stripping arbitrary whitespace from raw key bytes.
        handle.write(base64.urlsafe_b64encode(key))
        handle.flush()
        os.fsync(handle.fileno())


def verify_manifest(envelope: dict, trusted_public_key: str) -> dict:
    """Verify against an external trust pin, never an envelope's own untrusted key."""
    if envelope.get("public_key") != trusted_public_key:
        raise ValueError("Bundle signer does not match the trusted key")
    try:
        public = Ed25519PublicKey.from_public_bytes(base64.urlsafe_b64decode(trusted_public_key))
        public.verify(base64.urlsafe_b64decode(envelope["signature"]), rfc8785.dumps(envelope["manifest"]))
    except Exception as exc:
        raise ValueError("Bundle signature is invalid") from exc
    return deepcopy(envelope["manifest"])


def _artifact_paths(artifact_root: str | Path, files: dict[str, str], *, verify_hashes: bool = True) -> tuple:
    root = Path(artifact_root).resolve(strict=True)
    if not files or not root.is_dir():
        raise ValueError("Pinned model files and a local artifact directory are required")
    fingerprints = []
    for relative, expected in sorted(files.items()):
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts or str(path) != relative or "\\" in relative:
            raise ValueError("Artifact paths must be normalized relative paths")
        target = root.joinpath(*path.parts)
        # No symlink components: a post-validation redirection must not silently
        # turn a trusted checkpoint into another artifact outside its bundle.
        cursor = root
        for part in path.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise ValueError("Model artifact paths cannot traverse symlinks")
        if not target.is_file() or not target.resolve().is_relative_to(root):
            raise ValueError("Model artifact is missing or outside its root")
        if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError("Artifact hash must be lowercase SHA-256")
        before = target.stat()
        fingerprint = (relative, expected, before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        if verify_hashes:
            hasher = hashlib.sha256()
            with target.open("rb") as handle:
                for block in iter(lambda: handle.read(1024*1024), b""):
                    hasher.update(block)
            if hasher.hexdigest() != expected or target.stat() != before:
                raise ValueError("Model artifact hash mismatch or concurrent mutation")
        fingerprints.append(fingerprint)
    return (str(root), tuple(fingerprints))


class BundleRegistry:
    def __init__(self, store: LearningStore, signing_key_file: str | Path):
        self.store = store
        self._verified_files: dict[str, tuple] = {}
        self._key = Ed25519PrivateKey.from_private_bytes(base64.urlsafe_b64decode(read_secret(signing_key_file)))
        self.public_key = base64.urlsafe_b64encode(self._key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)).decode()

    def register(self, manifest: dict[str, Any], workspace_id: str, artifact_root: str | Path,
                 *, now: float | None = None) -> dict:
        current = time.time() if now is None else now
        manifest = deepcopy(manifest)
        required = ("id", "model_id", "model_revision", "files", "template_commitments", "languages", "calibrations", "synthetic")
        if any(k not in manifest for k in required) or not all(manifest[k] for k in required[:6]):
            raise ValueError("Bundle manifest is incomplete")
        if type(manifest["synthetic"]) is not bool or not isinstance(manifest["calibrations"], list):
            raise ValueError("Bundle synthetic flag and calibration list must be explicit")
        if manifest.get("workspace_id", workspace_id) != workspace_id:
            raise ValueError("Manifest workspace mismatch")
        if not isinstance(manifest["model_revision"], str) or not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", manifest["model_revision"]):
            raise ValueError("Model revision must be a full lowercase commit SHA or SHA-256 digest")
        if len(manifest["template_commitments"]) != len(set(manifest["template_commitments"])) or len(manifest["languages"]) != len(set(manifest["languages"])):
            raise ValueError("Bundle scopes must be unique")
        ids = set()
        bindings = set()
        for artifact in manifest["calibrations"]:
            validate_calibration(artifact)
            if artifact["model_bundle_id"] != manifest["id"] or artifact["template_commitment"] not in manifest["template_commitments"] or artifact["language"] not in manifest["languages"]:
                raise ValueError("Calibration is bound to another model or scope")
            binding = (artifact["template_commitment"], artifact["language"], artifact["question_id"])
            if artifact["id"] in ids or binding in bindings:
                raise ValueError("Duplicate calibration identity or scope")
            ids.add(artifact["id"])
            bindings.add(binding)
        verified_files = _artifact_paths(artifact_root, manifest["files"])
        manifest.update({"workspace_id": workspace_id, "registered_at": current, "schema_version": "rateloop.bundle.v1"})
        envelope = {"manifest": manifest, "public_key": self.public_key,
                    "signature": base64.urlsafe_b64encode(self._key.sign(rfc8785.dumps(manifest))).decode()}
        with self.store.transaction() as state:
            if manifest["id"] in state["bundles"]:
                raise ValueError("Bundle IDs are immutable")
            if manifest.get("snapshot_id"):
                LearningStore._validate_lineage(state, manifest["id"], workspace_id, current)
                if state["lineage"][manifest["id"]]["snapshot_id"] != manifest["snapshot_id"]:
                    raise ValueError("Bundle snapshot does not match its registered lineage")
            elif manifest["calibrations"]:
                raise ValueError("Calibrated bundles require training snapshot lineage")
            state["bundles"][manifest["id"]] = {"envelope": envelope, "artifact_root": str(Path(artifact_root).resolve())}
        self._verified_files[manifest["id"]] = verified_files
        return deepcopy(envelope)

    def _get(self, state: dict, bundle_id: str, workspace_id: str, now: float, *, verify_artifacts: bool = True) -> dict:
        record = state["bundles"].get(bundle_id)
        if not record:
            raise KeyError("Bundle not found")
        manifest = verify_manifest(record["envelope"], self.public_key)
        if manifest["workspace_id"] != workspace_id:
            raise KeyError("Bundle not found")
        if manifest.get("snapshot_id"):
            LearningStore._validate_lineage(state, bundle_id, workspace_id, now)
        fingerprints = _artifact_paths(record["artifact_root"], manifest["files"], verify_hashes=verify_artifacts)
        if verify_artifacts:
            self._verified_files[bundle_id] = fingerprints
        elif self._verified_files.get(bundle_id) != fingerprints:
            raise ValueError("Artifact verification cache is missing or files changed; reload the bundle")
        return {**deepcopy(record), "manifest": manifest}

    def get(self, bundle_id: str, workspace_id: str, *, now: float | None = None, verify_artifacts: bool = True) -> dict:
        with self.store.transaction() as state:
            return self._get(state, bundle_id, workspace_id, time.time() if now is None else now, verify_artifacts=verify_artifacts)

    @staticmethod
    def _quality_gate(manifest: dict, snapshot: dict, evidence: dict, *, now: float | None = None) -> dict:
        """Recompute decisions and human correctness from independent test groups.

        Evidence supplies raw scores, never its own correctness labels or a claimed
        aggregate. Exactly one deterministic representative per held-out group is
        audited to avoid treating repeated examples as independent trials.
        """
        current = time.time() if now is None else now
        observed_at, valid_until = evidence.get("observed_at"), evidence.get("valid_until")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in (observed_at, valid_until)):
            raise ValueError("Held-out evidence requires observed_at and valid_until timestamps")
        if not 0 < observed_at <= current < valid_until <= observed_at + 30*86400:
            raise PermissionError("Held-out evidence is expired, future-dated or exceeds the 30-day validation window")
        policy = manifest.get("selective_policy")
        if not isinstance(policy, dict) or set(policy) != {"threshold", "max_false_approval_rate", "minimum_coverage", "confidence"}:
            raise PermissionError("Selective operating policy must be pinned in the signed bundle before testing")
        if observed_at < manifest.get("registered_at", 0):
            raise PermissionError("Final test evidence must be observed after the candidate bundle and policy are registered")
        if manifest["synthetic"] or evidence.get("synthetic") is not False:
            raise PermissionError("Synthetic pilots cannot qualify real selective automation")
        if snapshot["purpose"] != "private_training":
            raise PermissionError("Selective deployment needs a private-training snapshot")
        template_commitment, language = evidence.get("template_commitment"), evidence.get("language")
        if template_commitment not in manifest["template_commitments"] or language not in manifest["languages"]:
            raise ValueError("Evidence scope does not match the bundle")
        threshold = evidence.get("threshold", .95)
        max_error = evidence.get("max_false_approval_rate", .01)
        min_coverage = evidence.get("minimum_coverage", .3)
        confidence = evidence.get("confidence", .95)
        if policy != {"threshold": threshold, "max_false_approval_rate": max_error, "minimum_coverage": min_coverage, "confidence": confidence}:
            raise ValueError("Test evidence differs from the predeclared selective policy")
        for value in (threshold, max_error, min_coverage, confidence):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < 1:
                raise ValueError("Gate thresholds must be probabilities strictly between zero and one")
        if confidence < .95 or max_error > .05 or threshold < .5:
            raise ValueError("Gate may not claim safety with weak confidence/error thresholds")
        train_groups = {r["group_id"] for r in snapshot["train"]}
        cal_groups = {r["group_id"] for r in snapshot["calibration"]}
        test_groups = {r["group_id"] for r in snapshot["test"]}
        if train_groups & cal_groups or train_groups & test_groups or cal_groups & test_groups:
            raise ValueError("Training, calibration and test groups overlap")
        calibration_by_question = {c["question_id"]: c for c in manifest["calibrations"]
                                   if c["template_commitment"] == template_commitment and c["language"] == language}
        representatives = {}
        for row in sorted(snapshot["test"], key=lambda r:r["evaluation_id"]):
            if row["template_commitment"] != template_commitment or row["template"].get("language") != language:
                raise ValueError("Test snapshot contains a different template or language")
            representatives.setdefault(row["group_id"], row)
        cases = {r["evaluation_id"]: r for r in representatives.values()}
        rows = evidence.get("rows", [])
        if not cases or not isinstance(rows, list) or len(rows) != len(cases) or {r.get("evaluation_id") for r in rows} != set(cases):
            raise ValueError("Evidence must cover every independent test-group representative exactly once")
        auto_approvals, wrong_approvals = 0, 0
        for observation in rows:
            row = cases[observation["evaluation_id"]]
            supplied = observation.get("raw_scores", {})
            questions = row["template"]["questions"]
            if set(supplied) != {q["id"] for q in questions}:
                raise ValueError("Test prediction must cover every question")
            approved, human_pass = True, True
            for question in questions:
                artifact = calibration_by_question.get(question["id"])
                if not artifact or set(artifact["example_ids"]) != cal_groups:
                    raise ValueError("Calibration must use exactly the independent calibration groups")
                probs = apply_temperature(supplied[question["id"]], artifact, model_bundle_id=manifest["id"],
                                          template_commitment=template_commitment, question_id=question["id"], language=language)
                label = max(probs, key=probs.get)
                approved = approved and label in question.get("passLabels", []) and probs[label] >= threshold
                human_pass = human_pass and row["labels"][question["id"]] in question.get("passLabels", [])
            auto_approvals += int(approved)
            wrong_approvals += int(approved and not human_pass)
        if auto_approvals == 0:
            raise PermissionError("No independently audited auto-approvals")
        bound = false_approval_upper_bound(wrong_approvals, auto_approvals, confidence)
        coverage = auto_approvals / len(cases)
        if bound > max_error or coverage < min_coverage:
            raise PermissionError("Held-out false-approval bound or automation coverage failed")
        return {"template_commitment": template_commitment, "language": language, "threshold": threshold,
                "observed_at": observed_at, "valid_until": valid_until,
                "auto_approvals": auto_approvals, "wrong_approvals": wrong_approvals, "test_groups": len(cases),
                "false_approval_upper_bound": bound, "confidence": confidence, "coverage": coverage,
                "max_false_approval_rate": max_error, "minimum_coverage": min_coverage,
                "calibration_ids": sorted(c["id"] for c in calibration_by_question.values()),
                "evidence_digest": hashlib.sha256(_json(evidence)).hexdigest()}

    def promote(self, bundle_id: str, workspace_id: str, *, template_commitment: str, language: str,
                mode: str = "shadow", evidence: dict | None = None, now: float | None = None) -> dict:
        if mode not in ("shadow", "assisted", "selective"):
            raise ValueError("Unsupported deployment mode")
        current = time.time() if now is None else now
        with self.store.transaction() as state:
            record = self._get(state, bundle_id, workspace_id, current)
            manifest = record["manifest"]
            if template_commitment not in manifest["template_commitments"] or language not in manifest["languages"]:
                raise ValueError("Promotion scope does not match the bundle")
            gate = None
            if mode == "selective":
                if not manifest.get("snapshot_id") or not evidence:
                    raise PermissionError("Selective promotion requires lineage and held-out evidence")
                if evidence.get("template_commitment") != template_commitment or evidence.get("language") != language:
                    raise ValueError("Promotion evidence scope mismatch")
                gate = self._quality_gate(manifest, state["snapshots"][manifest["snapshot_id"]], evidence, now=current)
            key = hashlib.sha256(_json([workspace_id, template_commitment, language])).hexdigest()
            previous = state["deployments"].get(key)
            deployment = {"workspace_id": workspace_id, "bundle_id": bundle_id, "template_commitment": template_commitment,
                          "language": language, "mode": mode, "promoted_at": current, "gate": gate,
                          "calibration_ids": sorted(c["id"] for c in manifest["calibrations"]
                                                    if c["template_commitment"] == template_commitment and c["language"] == language),
                          "previous": previous}
            state["deployments"][key] = deployment
            return deepcopy({k:v for k,v in deployment.items() if k != "previous"})

    def active(self, workspace_id: str, template_commitment: str, language: str, *, now: float | None = None, verify_artifacts: bool = True) -> dict:
        current = time.time() if now is None else now
        key = hashlib.sha256(_json([workspace_id, template_commitment, language])).hexdigest()
        with self.store.transaction() as state:
            deployment = state["deployments"].get(key)
            if not deployment:
                raise KeyError("No active bundle for this scope")
            self._get(state, deployment["bundle_id"], workspace_id, current, verify_artifacts=verify_artifacts)
            if deployment["mode"] == "selective" and (not deployment.get("gate") or deployment["gate"].get("valid_until", 0) <= current):
                raise PermissionError("Selective evidence expired; human review is required until revalidation")
            return deepcopy({k:v for k,v in deployment.items() if k != "previous"})

    def rollback(self, workspace_id: str, template_commitment: str, language: str, *, now: float | None = None) -> dict:
        current = time.time() if now is None else now
        key = hashlib.sha256(_json([workspace_id, template_commitment, language])).hexdigest()
        with self.store.transaction() as state:
            deployment = state["deployments"].get(key)
            if not deployment or not deployment.get("previous"):
                raise ValueError("No previous deployment is available")
            previous = deployment["previous"]
            self._get(state, previous["bundle_id"], workspace_id, current)
            if previous["mode"] == "selective" and (not previous.get("gate") or previous["gate"].get("valid_until", 0) <= current):
                raise PermissionError("Rollback cannot restore expired selective evidence")
            state["deployments"][key] = previous
            return deepcopy({k:v for k,v in previous.items() if k != "previous"})
