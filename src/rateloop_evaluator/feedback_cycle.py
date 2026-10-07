"""Import authenticated server feedback into fresh temporal private snapshots.

This is a generic worker contract, not a right to pool customer data. The trusted
connector attests human provenance; the worker enforces scope, lineage and splits.
"""
from copy import deepcopy
from datetime import datetime
import time
import uuid

from .learning import _digest, _source_aliases
from .protocol import commitment, validate_reference_demonstration_isolation

SCHEMA = "rateloop.evaluator.feedback-cycle.v1"


def _timestamp(value):
    if not isinstance(value, str):
        raise ValueError("Feedback timestamps must be explicit")
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def import_feedback_snapshot(store, *, workspace_id, template, model_bundle_id, rows, grant_id, request_id=None, now=None):
    current = time.time() if now is None else now
    if not isinstance(rows, list) or not 200 <= len(rows) <= 1000:
        raise ValueError("Feedback cycles need 200–1,000 fresh source groups")
    if len(template.questions) != 1:
        raise ValueError("Feedback cycles require one exact binary criterion")
    question = template.questions[0]
    tc = commitment(template.model_dump(), "rateloop.evaluator.template.v1")
    partitions = {key: [] for key in ("train", "calibration", "test")}
    fields = ["input.text", "input.context", "input.evidence"]
    with store.transaction() as state:
        request_key = _digest([workspace_id, request_id]) if request_id else None
        request_digest = _digest([template.model_dump(), model_bundle_id, rows, grant_id])
        previous = state.setdefault("feedback_cycle_requests", {}).get(request_key) if request_key else None
        if previous:
            if previous["digest"] != request_digest:
                raise ValueError("A feedback cycle request changed immutable content")
            snapshot = state["snapshots"][previous["snapshot_id"]]
            store._validate_snapshot(state, snapshot, workspace_id, current)
            return deepcopy(snapshot)
        used = state.setdefault("feedback_cycle_used_aliases", {}).setdefault(workspace_id, {})
        aliases = set()
        for row in rows:
            expected = {"caseId", "sourceGroupId", "input", "labels", "inputCommitment", "createdAt", "observedAt", "lineage", "evidenceFingerprint", "partition"}
            if not isinstance(row, dict) or set(row) != expected or row["partition"] not in partitions:
                raise ValueError("Feedback row contract mismatch")
            if (set(row["input"]) != {"text", "context", "evidence"}
                    or any(not isinstance(v, str) for v in row["input"].values())
                    or not row["input"]["text"].strip()
                    or set(row["labels"]) != {question.id}
                    or row["labels"][question.id] not in {label.id for label in question.labels}):
                raise ValueError("Feedback content does not match the exact criterion")
            lineage = row["lineage"]
            if (not isinstance(lineage, dict) or lineage.get("aiExposed") is not False
                    or not lineage.get("humanResultCommitment") or not lineage.get("auditIds")
                    or lineage.get("templateCommitment") != tc
                    or lineage.get("inputCommitment") != row["inputCommitment"]
                    or not isinstance(lineage.get("modelBundleId"), str) or not lineage["modelBundleId"]
                    or lineage.get("trainingModelBundleId") != model_bundle_id
                    or commitment(lineage, "rateloop.evaluator.feedback-evidence.v1") != row["evidenceFingerprint"]):
                raise ValueError("Authenticated independent feedback lineage is missing or inconsistent")
            created, observed = _timestamp(row["createdAt"]), _timestamp(row["observedAt"])
            if not created <= observed <= current:
                raise ValueError("Feedback time ordering is invalid")
            grants = store._matching_grants(state, workspace_id=workspace_id, right="private_training",
                case_id=row["caseId"], template_id=template.id, fields=[*fields, "human_labels"], now=current,
                model_bundle_id=model_bundle_id, template_commitment=tc)
            if grant_id not in grants:
                raise PermissionError("Feedback was not covered by the leased dataset permission")
            example = {"evaluation_id": "feedback_"+_digest([workspace_id, row["caseId"], row["evidenceFingerprint"]]),
                "workspace_id": workspace_id, "case_id": row["caseId"], "group_id": row["sourceGroupId"],
                "source_group_id": row["sourceGroupId"], "template": template.model_dump(), "template_commitment": tc,
                "model_bundle_id": model_bundle_id, "input_commitment": row["inputCommitment"],
                "input": deepcopy(row["input"]), "labels": deepcopy(row["labels"]), "fields": fields,
                "grant_ids": [grant_id], "created_at": created, "source_kind": "evaluation",
                "human_labels": [{"independent_human": True, "exposed_to_ai": False, "quarantine_reasons": [],
                    "labels": deepcopy(row["labels"]), "source_evidence": deepcopy(lineage)}]}
            keys = {_digest(alias) for alias in _source_aliases(example)}
            if keys & (aliases | set(used)):
                raise ValueError("A feedback source was duplicated or used by a previous cycle")
            aliases.update(keys)
            partitions[row["partition"]].append(example)
        for partition, examples in partitions.items():
            if len(examples) < (30 if partition == "test" else 1):
                raise ValueError("Feedback partitions need independent source groups")
            for label in question.labels:
                if sum(row["labels"][question.id] == label.id for row in examples) < 10:
                    raise ValueError("Feedback partitions need both labels")
        if not (max(row["created_at"] for row in partitions["train"]) < min(row["created_at"] for row in partitions["calibration"])
                and max(row["created_at"] for row in partitions["calibration"]) < min(row["created_at"] for row in partitions["test"])):
            raise ValueError("Final feedback must be strictly later than training and calibration")
        validate_reference_demonstration_isolation(row for part in partitions.values() for row in part)
        snapshot = {"id": "snapshot_"+uuid.uuid4().hex, "workspace_id": workspace_id,
            "template_id": template.id, "template_version": template.version, "purpose": "private_training",
            "created_at": current, "seed": SCHEMA, "grant_ids": [grant_id], "invalidated_at": None,
            "group_count": len(rows), **partitions}
        snapshot["content_digest"] = _digest({key: snapshot[key] for key in ("workspace_id", "template_id", "template_version", "purpose", "train", "calibration", "test")})
        state["snapshots"][snapshot["id"]] = snapshot
        if request_key:
            state["feedback_cycle_requests"][request_key] = {"digest": request_digest, "snapshot_id": snapshot["id"]}
        for alias in aliases:
            used[alias] = snapshot["id"]
        return deepcopy(snapshot)


class TrainingMajorityReference:
    """A deterministic reference selected solely from the training partition."""
    def __init__(self, snapshot):
        self.labels = {}
        for question in snapshot["train"][0]["template"]["questions"]:
            labels = [label["id"] for label in question["labels"]]
            counts = {label: sum(row["labels"][question["id"]] == label for row in snapshot["train"]) for label in labels}
            self.labels[question["id"]] = max(labels, key=lambda label: counts[label])
    def count_tokens(self, text, questions):
        return 0
    def predict(self, text, questions):
        return {q["id"]: {label["id"]: float(label["id"] == self.labels[q["id"]]) for label in q["labels"]} for q in questions}


def erase_feedback_permission(store, *, workspace_id, grant_id, now=None):
    """Remove retained examples on withdrawal; preserve hashed split tombstones."""
    current = time.time() if now is None else now
    with store.transaction() as state:
        affected = set()
        for snapshot_id, snapshot in state["snapshots"].items():
            if snapshot["workspace_id"] == workspace_id and snapshot.get("seed") == SCHEMA and grant_id in snapshot["grant_ids"]:
                snapshot.update(invalidated_at=current, invalidation_reason="source_erased")
                for partition in ("train", "calibration", "test"):
                    snapshot[partition] = []
                affected.add(snapshot_id)
        for lineage in state["lineage"].values():
            if lineage["workspace_id"] == workspace_id and lineage["snapshot_id"] in affected:
                lineage.update(retired_at=current, retirement_reason="source_erased")
