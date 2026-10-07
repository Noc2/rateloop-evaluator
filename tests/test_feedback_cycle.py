from copy import deepcopy
from datetime import datetime, timezone
import time
import pytest
from rateloop_evaluator.feedback_cycle import import_feedback_snapshot, TrainingMajorityReference
from rateloop_evaluator.learning import LearningStore, provision_key, is_independent_reference
from rateloop_evaluator.protocol import commitment
from rateloop_evaluator.templates import custom_text_evaluation


def iso(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat().replace("+00:00", "Z")


@pytest.fixture
def setup(tmp_path):
    key = tmp_path / "key"; provision_key(key)
    store = LearningStore(tmp_path / "data", key)
    template = custom_text_evaluation("en", "Does the text follow the instruction?", "Yes", "No")
    tc = commitment(template.model_dump(), "rateloop.evaluator.template.v1")
    now = time.time()
    grant = store.add_grant(workspace_id="w", rights=["private_training"], expires_at=now+3600,
        evidence="Explicit authenticated owner permission", fields=["input.text", "input.context", "input.evidence", "human_labels"],
        model_bundle_ids=["base"], template_commitments=[tc])
    rows = []
    for i in range(200):
        lineage = {"aiExposed": False, "humanResultCommitment": f"human-{i}", "auditIds": [f"audit-{i}"],
            "templateCommitment": tc, "inputCommitment": f"input-{i}", "modelBundleId": "source-v2", "trainingModelBundleId": "base"}
        rows.append({"caseId": f"case-{i}", "sourceGroupId": f"group-{i}", "input": {"text": f"Sample {i}", "context": "", "evidence": ""},
            "labels": {"judgment": "approved" if i % 2 else "rejected"}, "inputCommitment": f"input-{i}",
            "createdAt": iso(now-1000+i), "observedAt": iso(now-999+i), "lineage": lineage,
            "evidenceFingerprint": commitment(lineage, "rateloop.evaluator.feedback-evidence.v1"),
            "partition": "train" if i < 140 else "calibration" if i < 170 else "test"})
    return store, dict(workspace_id="w", template=template, model_bundle_id="base", rows=rows, grant_id=grant["id"], now=now)


def test_snapshot_is_independent_temporal_scoped_and_revocable(setup):
    store, args = setup
    snapshot = import_feedback_snapshot(store, **args)
    assert [len(snapshot[p]) for p in ("train", "calibration", "test")] == [140, 30, 30]
    assert all(is_independent_reference(row) for row in snapshot["test"])
    assert TrainingMajorityReference(snapshot).predict("", snapshot["test"][0]["template"]["questions"])
    with pytest.raises(ValueError, match="previous cycle"):
        import_feedback_snapshot(store, **args)
    store.revoke_grant(args["grant_id"], "w")
    with pytest.raises(PermissionError):
        store.load_snapshot(snapshot["id"], "w")


def test_interrupted_cycle_import_replays_only_exact_immutable_request(setup):
    store, args = setup
    first = import_feedback_snapshot(store, **args, request_id="cycle-1")
    assert import_feedback_snapshot(store, **args, request_id="cycle-1")["id"] == first["id"]
    changed = deepcopy(args)
    changed["rows"][0]["input"]["text"] = "Changed bytes"
    with pytest.raises(ValueError, match="immutable"):
        import_feedback_snapshot(store, **changed, request_id="cycle-1")


@pytest.mark.parametrize("mutation", ["exposed", "wrong_target", "tampered_lineage", "temporal", "duplicate", "labels", "workspace"])
def test_invalid_feedback_cannot_enter_a_snapshot(setup, mutation):
    store, original = setup
    args = deepcopy(original)
    rows = args["rows"]
    if mutation == "exposed":
        rows[0]["lineage"]["aiExposed"] = True
        rows[0]["evidenceFingerprint"] = commitment(rows[0]["lineage"], "rateloop.evaluator.feedback-evidence.v1")
    if mutation == "wrong_target":
        rows[0]["lineage"]["trainingModelBundleId"] = "other"
        rows[0]["evidenceFingerprint"] = commitment(rows[0]["lineage"], "rateloop.evaluator.feedback-evidence.v1")
    if mutation == "tampered_lineage": rows[0]["lineage"]["humanResultCommitment"] = "changed"
    if mutation == "temporal": rows[170]["createdAt"] = rows[0]["createdAt"]
    if mutation == "duplicate": rows[199]["input"] = rows[0]["input"]
    if mutation == "labels":
        for row in rows: row["labels"]["judgment"] = "approved"
    if mutation == "workspace": args["workspace_id"] = "other"
    with pytest.raises((ValueError, PermissionError)):
        import_feedback_snapshot(store, **args)
    with store.transaction() as state: assert not state["snapshots"]
