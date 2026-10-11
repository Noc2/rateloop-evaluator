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


def modern_args(args):
    value = deepcopy(args)
    for row in value["rows"]:
        row["replayed"] = False
        row["lineage"]["sourceContentHash"] = "source-" + row["caseId"]
        row["evidenceFingerprint"] = commitment(row["lineage"], "rateloop.evaluator.feedback-evidence.v1")
    return value


def fresh_cycle(args):
    value = deepcopy(args)
    for index, row in enumerate(value["rows"]):
        suffix = f"fresh-{index}"
        row.update(caseId=suffix, sourceGroupId="group-" + suffix, inputCommitment="input-" + suffix,
                   input={"text": "Fresh source " + suffix, "context": "", "evidence": ""}, replayed=False,
                   createdAt=iso(value["now"] - 600 + index), observedAt=iso(value["now"] - 599 + index))
        row["lineage"].update(inputCommitment=row["inputCommitment"], humanResultCommitment="human-" + suffix,
                             auditIds=["audit-" + suffix], sourceContentHash="source-" + suffix)
        row["evidenceFingerprint"] = commitment(row["lineage"], "rateloop.evaluator.feedback-evidence.v1")
    return value


def with_replay(first, next_args, indexes=range(40)):
    value = deepcopy(next_args)
    value["rows"] = [dict(deepcopy(first["rows"][i]), partition="train", replayed=True) for i in indexes] + value["rows"]
    return value


def test_training_only_replay_keeps_original_time_fingerprint_and_inner_role(setup):
    from rateloop_evaluator.quality import validation_partition
    store, original = setup
    args = modern_args(original)
    first = import_feedback_snapshot(store, **args, request_id="first")
    following = with_replay(args, fresh_cycle(args))
    second = import_feedback_snapshot(store, **following, request_id="second")
    assert second["seed"] == "rateloop.evaluator.feedback-cycle.v2"
    assert [len(second[p]) for p in ("train", "calibration", "test")] == [180, 30, 30]
    original_by_case = {row["case_id"]: row for row in first["train"]}
    for row in second["train"][:40]:
        assert row == original_by_case[row["case_id"]]
    def roles(snapshot):
        train, validation = validation_partition(snapshot)
        return {row["group_id"]: role for role, rows in (("train", train), ("validation", validation)) for row in rows}
    before, after = roles(first), roles(second)
    assert all(before[group] == after[group] for group in before.keys() & after.keys())
    assert import_feedback_snapshot(store, **following, request_id="second")["id"] == second["id"]


@pytest.mark.parametrize("mutation", ["heldout", "legacy", "fingerprint", "group", "input", "time", "not_train", "unknown", "source_alias", "fresh_duplicate", "too_few_fresh", "mixed_contract"])
def test_replay_cannot_reuse_holdouts_or_change_immutable_source_identity(setup, mutation):
    store, original = setup
    args = modern_args(original)
    if mutation == "legacy":
        for row in args["rows"]: row.pop("replayed")
    import_feedback_snapshot(store, **args)
    follow = with_replay(args, fresh_cycle(args))
    row = follow["rows"][0]
    if mutation == "heldout":
        follow["rows"][0] = dict(deepcopy(args["rows"][175]), partition="train", replayed=True)
    elif mutation == "fingerprint":
        row["lineage"]["humanResultCommitment"] = "changed"
        row["evidenceFingerprint"] = commitment(row["lineage"], "rateloop.evaluator.feedback-evidence.v1")
    elif mutation == "group": row["sourceGroupId"] = "changed"
    elif mutation == "input": row["input"]["text"] += "changed"
    elif mutation == "time": row["observedAt"] = iso(follow["now"])
    elif mutation == "not_train": row["partition"] = "calibration"
    elif mutation == "unknown": row["caseId"] = "unknown"
    elif mutation == "source_alias":
        row["lineage"]["sourceContentHash"] = args["rows"][175]["lineage"]["sourceContentHash"]
        row["evidenceFingerprint"] = commitment(row["lineage"], "rateloop.evaluator.feedback-evidence.v1")
    elif mutation == "fresh_duplicate":
        row = follow["rows"][40]
        row["lineage"]["sourceContentHash"] = args["rows"][175]["lineage"]["sourceContentHash"]
        row["evidenceFingerprint"] = commitment(row["lineage"], "rateloop.evaluator.feedback-evidence.v1")
    elif mutation == "too_few_fresh": follow["rows"].pop()
    elif mutation == "mixed_contract": row.pop("replayed")
    with store.transaction() as state:
        before = deepcopy(state)
    with pytest.raises((ValueError, PermissionError)):
        import_feedback_snapshot(store, **follow)
    with store.transaction() as state:
        assert state == before, "Rejected replay must not consume any source or partially rewrite tombstones"


def test_withdrawal_erases_both_versions_but_regrant_cannot_reset_holdout_tombstones(setup):
    from rateloop_evaluator.feedback_cycle import erase_feedback_permission
    store, original = setup
    args = modern_args(original)
    first = import_feedback_snapshot(store, **args)
    store.revoke_grant(args["grant_id"], "w")
    erase_feedback_permission(store, workspace_id="w", grant_id=args["grant_id"], now=args["now"])
    with pytest.raises(PermissionError):
        import_feedback_snapshot(store, **with_replay(args, fresh_cycle(args)))
    grant = store.add_grant(workspace_id="w", rights=["private_training"], expires_at=args["now"] + 3600,
        evidence="New explicit authenticated owner permission", fields=["input.text", "input.context", "input.evidence", "human_labels"],
        model_bundle_ids=["base"], template_commitments=[commitment(args["template"].model_dump(), "rateloop.evaluator.template.v1")])
    following = with_replay(args, fresh_cycle(args))
    following["grant_id"] = grant["id"]
    forbidden = with_replay(args, fresh_cycle(args), indexes=[175])
    forbidden["grant_id"] = grant["id"]
    with pytest.raises(ValueError, match="previous training"):
        import_feedback_snapshot(store, **forbidden)
    second = import_feedback_snapshot(store, **following)
    assert second["grant_ids"] == [grant["id"]]
    with store.transaction() as state:
        assert state["snapshots"][first["id"]]["train"] == []
        assert state["snapshots"][first["id"]]["invalidated_at"] is not None
        for record in state["feedback_cycle_used_aliases"]["w"].values():
            assert "caseId" not in record
            assert len(record["caseDigest"]) == 64
            assert len(record["identityDigest"]) == 64


def test_legacy_snapshot_erasure_retains_unknown_role_tombstones(setup):
    from rateloop_evaluator.feedback_cycle import erase_feedback_permission
    store, args = setup
    legacy = import_feedback_snapshot(store, **args)
    assert legacy["seed"] == "rateloop.evaluator.feedback-cycle.v1"
    with store.transaction() as state:
        tombstones = deepcopy(state["feedback_cycle_used_aliases"])
    erase_feedback_permission(store, workspace_id="w", grant_id=args["grant_id"], now=args["now"])
    with store.transaction() as state:
        assert state["snapshots"][legacy["id"]]["train"] == []
        assert state["feedback_cycle_used_aliases"] == tombstones


def test_replay_limit_and_conflicting_prior_alias_fail_closed(setup):
    from rateloop_evaluator.learning import _digest
    store, original = setup
    args = modern_args(original)
    import_feedback_snapshot(store, **args)
    oversized = with_replay(args, fresh_cycle(args), indexes=[0] * 501)
    with pytest.raises(ValueError, match="at most 500"):
        import_feedback_snapshot(store, **oversized)
    aliases_key = _digest("case:" + args["rows"][0]["caseId"])
    with store.transaction() as state:
        ledger = state["feedback_cycle_used_aliases"]["w"]
        ledger[aliases_key]["partition"] = "test"
        protected = deepcopy(ledger)
    with pytest.raises(ValueError, match="previous training"):
        import_feedback_snapshot(store, **with_replay(args, fresh_cycle(args)))
    with store.transaction() as state:
        assert state["feedback_cycle_used_aliases"]["w"] == protected
