from copy import deepcopy
from pathlib import Path
import time

import pytest

from rateloop_evaluator.learning import LearningStore, provision_key

TEMPLATE = {"id": "reply", "version": "1", "language": "en", "questions": [
    {"id": "supported", "text": "Is the reply supported?", "labels": [
        {"id": "yes", "description": "Supported"}, {"id": "no", "description": "Unsupported"}], "passLabels": ["yes"]}]}
FIELDS = ["input.text", "input.context", "input.evidence"]


@pytest.fixture
def store(tmp_path):
    key = tmp_path / "customer.key"
    provision_key(key)
    return LearningStore(tmp_path / "private", key)


def grant(store, workspace="workspace-a", rights=None, **kwargs):
    return store.add_grant(workspace_id=workspace, rights=rights or ["ai_use", "private_training"],
                           expires_at=kwargs.pop("expires_at", time.time()+3600),
                           evidence="Explicit local owner approval", **kwargs)


def example(store, number, *, workspace="workspace-a", group_id=None, text=None):
    row = store.record_evaluation(evaluation_id=f"eval-{workspace}-{number}", workspace_id=workspace,
        case_id=f"case-{number}", input_commitment=f"input-{number}", template_commitment="template-sha",
        template=TEMPLATE, input_payload={"text": text or f"Private example {number}", "context": "Sensitive context", "evidence": []},
        group_id=group_id, fields=FIELDS)
    feedback = store.add_feedback(workspace_id=workspace, evaluation_id=row["evaluation_id"],
        input_commitment=row["input_commitment"], template_commitment=row["template_commitment"],
        annotator_id="reviewer-1", labels={"supported": "yes"}, exposed_to_ai=False, independent_human=True)
    return row, feedback


def snapshot(store, count=6):
    for i in range(count):
        example(store, i)
    return store.create_snapshot("workspace-a", "reply", "1")


def test_encrypted_at_rest_and_customer_key_required(store):
    grant(store)
    row, _ = example(store, 1)
    for path in store.root.iterdir():
        assert b"Sensitive context" not in path.read_bytes()
        assert b"workspace-a" not in path.read_bytes()
        assert b"reviewer-1" not in path.read_bytes()
    assert row["input"]["context"] == "Sensitive context"
    assert store._path.stat().st_mode & 0o077 == 0


def test_keys_are_exclusive_and_not_world_readable(tmp_path):
    key = tmp_path / "key"
    provision_key(key)
    with pytest.raises(FileExistsError):
        provision_key(key)
    key.chmod(0o644)
    with pytest.raises(ValueError, match="only to its owner"):
        LearningStore(tmp_path / "data", key)


def test_rights_workspace_case_template_fields_and_expiry_are_independent(store):
    grant(store, rights=["ai_use"], case_ids=["case-1"], template_ids=["reply"], fields=["input.text"])
    scope = dict(workspace_id="workspace-a", right="ai_use", case_id="case-1", template_id="reply", fields=["input.text"])
    assert store.check_right(**scope)
    for changes in ({"workspace_id":"workspace-b"}, {"right":"private_training"},
                    {"right":"shared_contribution"}, {"right":"public_weight_distribution"},
                    {"case_id":"case-2"}, {"template_id":"different"}, {"fields":["input.context"]},
                    {"now":time.time()+7200}):
        with pytest.raises(PermissionError):
            store.check_right(**{**scope, **changes})


def test_ai_use_does_not_authorize_raw_retention(store):
    grant(store, rights=["ai_use"])
    with pytest.raises(PermissionError, match="private_training"):
        example(store, 1)
    row = store.record_evaluation(evaluation_id="metadata",workspace_id="workspace-a",case_id="1",
        input_commitment="input",template_commitment="template",template=TEMPLATE)
    assert row["input"] is None


def test_retained_fields_cannot_be_omitted_from_scope(store):
    grant(store, fields=["input.text"])
    with pytest.raises(ValueError, match="Every retained input field"):
        store.record_evaluation(evaluation_id="e", workspace_id="workspace-a", case_id="1", input_commitment="i",
            template_commitment="t", template=TEMPLATE, input_payload={"text":"x","context":"secret"},fields=["input.text"])


def test_feedback_binding_exposure_duplicates_and_tenant_isolation(store):
    grant(store)
    row, original = example(store, 1)
    args = dict(workspace_id="workspace-a",evaluation_id=row["evaluation_id"],input_commitment=row["input_commitment"],
                template_commitment=row["template_commitment"], annotator_id="other-reviewer", labels={"supported":"yes"},
                exposed_to_ai=False, independent_human=True)
    mismatch = store.add_feedback(**{**args,"input_commitment":"wrong"})
    assert "commitment_mismatch" in mismatch["quarantine_reasons"]
    duplicate = store.add_feedback(**{**args,"annotator_id":"reviewer-1"})
    assert "duplicate_reviewer" in duplicate["quarantine_reasons"]
    exposed = store.add_feedback(**{**args,"annotator_id":"exposed-reviewer","exposed_to_ai":True})
    assert "not_independent_human" in exposed["quarantine_reasons"]
    with pytest.raises(KeyError):
        store.add_feedback(**{**args,"workspace_id":"workspace-b"})
    assert not original["quarantine_reasons"]


def test_grouped_partitions_keep_duplicates_and_source_groups_together(store):
    grant(store)
    for n in range(10):
        example(store,n,group_id="shared-source" if n < 2 else None,text="Exact duplicate" if n in (2,3) else None)
    snap = store.create_snapshot("workspace-a","reply","1")
    partitions = {part: {r["evaluation_id"] for r in snap[part]} for part in ("train","calibration","test")}
    assert all(partitions.values())
    for a,b in ((0,1),(2,3)):
        assert any({f"eval-workspace-a-{a}",f"eval-workspace-a-{b}"} <= ids for ids in partitions.values())
    groups = [{r["group_id"] for r in snap[part]} for part in partitions]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert snap == store.load_snapshot(snap["id"],"workspace-a")
    with pytest.raises(KeyError):
        store.load_snapshot(snap["id"],"workspace-b")


def test_revocation_invalidates_snapshot_and_retires_derived_model(store):
    g = grant(store)
    snap = snapshot(store)
    store.register_model_lineage("bundle",snap["id"],"workspace-a")
    store.assert_model_usable("bundle","workspace-a")
    impact = store.revoke_grant(g["id"],"workspace-a")
    assert impact["invalidated_snapshots"] == [snap["id"]]
    assert impact["retired_bundles"] == ["bundle"]
    with pytest.raises(PermissionError):
        store.load_snapshot(snap["id"],"workspace-a")
    with pytest.raises(PermissionError):
        store.assert_model_usable("bundle","workspace-a")
    grant(store)
    with pytest.raises(PermissionError):
        store.load_snapshot(snap["id"],"workspace-a")


def test_expiry_invalidates_snapshot_even_if_new_grant_exists(store):
    expires = time.time()+100
    grant(store, expires_at=expires)
    snap = snapshot(store)
    grant(store, expires_at=expires+1000)
    with pytest.raises(PermissionError,match="expired or revoked"):
        store.load_snapshot(snap["id"],"workspace-a",now=expires+1)


def test_disagreement_retains_labels_and_invalidates_prior_snapshot(store):
    grant(store)
    snap = snapshot(store)
    row = snap["train"][0]
    conflict = store.add_feedback(workspace_id="workspace-a", evaluation_id=row["evaluation_id"],
        input_commitment=row["input_commitment"],template_commitment=row["template_commitment"],
        annotator_id="reviewer-2",labels={"supported":"no"},exposed_to_ai=False,independent_human=True)
    assert "human_disagreement" in conflict["quarantine_reasons"]
    with pytest.raises(PermissionError):
        store.load_snapshot(snap["id"],"workspace-a")
    with store.transaction() as state:
        related = [f for f in state["feedback"].values() if f["evaluation_id"] == row["evaluation_id"]]
    assert len(related) == 2
    assert {f["labels"]["supported"] for f in related} == {"yes","no"}
    assert all("human_disagreement" in f["quarantine_reasons"] for f in related)


def test_shared_contribution_does_not_grant_public_weight_distribution(store):
    grant(store, rights=["ai_use","private_training","shared_contribution"])
    snapshot(store)
    assert store.create_snapshot("workspace-a","reply","1",purpose="shared_contribution")
    with pytest.raises(ValueError,match="Insufficient independent"):
        store.create_snapshot("workspace-a","reply","1",purpose="public_weight_distribution")


def test_delete_removes_raw_content_and_invalidates_models(store):
    grant(store)
    snap = snapshot(store)
    row = snap["train"][0]
    store.register_model_lineage("bundle",snap["id"],"workspace-a")
    report = store.delete_case("workspace-a",row["case_id"])
    assert report["deleted_evaluations"] == 1
    with pytest.raises(PermissionError):
        store.assert_model_usable("bundle","workspace-a")
    with store.transaction() as state:
        assert row["evaluation_id"] not in state["evaluations"]
        assert all(f["evaluation_id"] != row["evaluation_id"] for f in state["feedback"].values())
        assert all(r["evaluation_id"] != row["evaluation_id"] for s in state["snapshots"].values() for part in ("train","calibration","test") for r in s[part])



def test_public_weight_release_is_independent_of_sharing_raw_examples(store):
    grant(store, rights=["ai_use","private_training","public_weight_distribution"])
    snapshot(store)
    assert store.create_snapshot("workspace-a","reply","1",purpose="public_weight_distribution")
    with pytest.raises(ValueError,match="Insufficient independent"):
        store.create_snapshot("workspace-a","reply","1",purpose="shared_contribution")


def test_exact_model_and_template_commitment_scopes_are_enforced_through_training(store):
    g=grant(store,model_bundle_ids=["base-model"],template_commitments=["template-sha"])
    scope=dict(workspace_id="workspace-a",right="ai_use",case_id="case-1",template_id="reply",fields=FIELDS,
               model_bundle_id="base-model",template_commitment="template-sha")
    assert store.check_right(**scope)
    for change in ({"model_bundle_id":"other"},{"template_commitment":"changed"},{"model_bundle_id":None}):
        with pytest.raises(PermissionError):
            store.check_right(**{**scope,**change})
    for n in range(3):
        row=store.record_evaluation(evaluation_id=f"scoped-{n}",workspace_id="workspace-a",case_id=f"case-{n}",
            input_commitment=f"input-{n}",template_commitment="template-sha",template=TEMPLATE,
            input_payload={"text":f"text-{n}","context":"","evidence":""},model_bundle_id="base-model")
        store.add_feedback(workspace_id="workspace-a",evaluation_id=row["evaluation_id"],input_commitment=row["input_commitment"],
            template_commitment="template-sha",annotator_id="reviewer",labels={"supported":"yes"},
            exposed_to_ai=False,independent_human=True)
    snap=store.create_snapshot("workspace-a","reply","1")
    assert store.load_snapshot(snap["id"],"workspace-a")
    store.revoke_grant(g["id"],"workspace-a")
    with pytest.raises(PermissionError):
        store.load_snapshot(snap["id"],"workspace-a")


def test_consent_changes_do_not_change_committed_evaluation_identity(store):
    grant(store,rights=["ai_use"])
    args=dict(evaluation_id="repeat",workspace_id="workspace-a",case_id="case-1",input_commitment="input",
              template_commitment="template-sha",template=TEMPLATE)
    assert store.record_evaluation(**args)["input"] is None
    private=grant(store,rights=["private_training"])
    submitted={"text":"Explicitly resubmitted content","context":"","evidence":""}
    assert store.record_evaluation(**args,input_payload=submitted)["input"] == submitted
    with pytest.raises(ValueError,match="different retained content"):
        store.record_evaluation(**args,input_payload={**submitted,"text":"different"})
    store.revoke_grant(private["id"],"workspace-a")
    assert store.record_evaluation(**args)["input"] is None
    with pytest.raises(PermissionError):
        store.record_evaluation(**args,input_payload=submitted)
    with pytest.raises(ValueError,match="different content"):
        store.record_evaluation(**{**args,"input_commitment":"changed"})
