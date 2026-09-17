from copy import deepcopy
import hashlib
import time
import pytest

from rateloop_evaluator.calibration import fit_temperature
from rateloop_evaluator.learning import LearningStore, provision_key
from rateloop_evaluator.registry import BundleRegistry, provision_signing_key, verify_manifest


@pytest.fixture
def setup_registry(tmp_path):
    encryption = tmp_path / "encryption.key"
    signing = tmp_path / "signing.key"
    provision_key(encryption)
    provision_signing_key(signing)
    store = LearningStore(tmp_path / "store", encryption)
    registry = BundleRegistry(store,signing)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "weights.bin").write_bytes(b"test-checkpoint-fixture")
    manifest = {"id":"bundle-1","model_id":"fixture-test-only","model_revision":"a"*40,
        "files":{"weights.bin":hashlib.sha256(b"test-checkpoint-fixture").hexdigest()},
        "template_commitments":["template"],"languages":["en"],"calibrations":[],"synthetic":True}
    return store,registry,model_dir,manifest


def test_signed_manifest_requires_pinned_trust_and_detects_tampering(setup_registry):
    store,registry,model_dir,manifest=setup_registry
    envelope=registry.register(manifest,"workspace",model_dir)
    assert verify_manifest(envelope,registry.public_key)["id"] == "bundle-1"
    with pytest.raises(ValueError,match="trusted key"):
        verify_manifest(envelope,"attacker-key")
    envelope["manifest"]["model_id"]="tampered"
    with pytest.raises(ValueError,match="signature"):
        verify_manifest(envelope,registry.public_key)
    with pytest.raises(ValueError,match="immutable"):
        registry.register(manifest,"workspace",model_dir)


def test_artifact_hash_paths_cache_and_tenant_isolation(setup_registry):
    _,registry,model_dir,manifest=setup_registry
    registry.register(manifest,"workspace",model_dir)
    assert registry.get("bundle-1","workspace",verify_artifacts=False)["manifest"]["id"] == "bundle-1"
    with pytest.raises(KeyError):
        registry.get("bundle-1","another-workspace")
    (model_dir / "weights.bin").write_bytes(b"modified")
    with pytest.raises(ValueError,match="cache is missing or files changed"):
        registry.get("bundle-1","workspace",verify_artifacts=False)
    with pytest.raises(ValueError,match="hash mismatch"):
        registry.get("bundle-1","workspace")
    manifest["id"]="escape"
    manifest["files"]={"../elsewhere": "0"*64}
    with pytest.raises(ValueError,match="normalized relative"):
        registry.register(manifest,"workspace",model_dir)


def test_shadow_promotion_and_rollback_preserve_modes(setup_registry):
    _,registry,model_dir,manifest=setup_registry
    registry.register(manifest,"workspace",model_dir)
    registry.promote("bundle-1","workspace",template_commitment="template",language="en",mode="shadow")
    manifest["id"]="bundle-2"
    registry.register(manifest,"workspace",model_dir)
    registry.promote("bundle-2","workspace",template_commitment="template",language="en",mode="assisted")
    assert registry.active("workspace","template","en")["bundle_id"] == "bundle-2"
    assert registry.rollback("workspace","template","en")["bundle_id"] == "bundle-1"
    assert registry.active("workspace","template","en")["mode"] == "shadow"
    with pytest.raises(PermissionError,match="lineage and held-out"):
        registry.promote("bundle-1","workspace",template_commitment="template",language="en",mode="selective")


def gate_inputs(count=300):
    calibration=fit_temperature([{"yes":.95,"no":.05}]*2,["yes"]*2,model_bundle_id="model",
        template_commitment="template",question_id="q",language="en",example_ids=["cal-a","cal-b"])
    template={"language":"en","questions":[{"id":"q","passLabels":["yes"]}]}
    test=[{"evaluation_id":f"test-{i:04d}","group_id":f"test-group-{i}","template_commitment":"template",
           "template":template,"labels":{"q":"yes"}} for i in range(count)]
    snapshot={"purpose":"private_training","train":[{"group_id":"train-a"}],
              "calibration":[{"group_id":"cal-a"},{"group_id":"cal-b"}],"test":test}
    manifest={"id":"model","synthetic":False,"template_commitments":["template"],"languages":["en"],"calibrations":[calibration],
              "selective_policy":{"threshold":.95,"max_false_approval_rate":.01,"minimum_coverage":.3,"confidence":.95}}
    evidence={"synthetic":False,"template_commitment":"template","language":"en","threshold":.95,
              "observed_at":time.time(),"valid_until":time.time()+86400,
              "rows":[{"evaluation_id":row["evaluation_id"],"raw_scores":{"q":{"yes":.95,"no":.05}}} for row in test]}
    return manifest,snapshot,evidence


def test_gate_recomputes_independent_human_correctness_and_exact_bound():
    manifest,snapshot,evidence=gate_inputs()
    report=BundleRegistry._quality_gate(manifest,snapshot,evidence)
    assert report["auto_approvals"]==300
    assert report["wrong_approvals"]==0
    assert report["false_approval_upper_bound"] < .01
    snapshot["test"][0]["labels"]["q"]="no"
    # A caller's invented aggregate cannot override retained human truth.
    evidence["wrong_approvals"]=0
    with pytest.raises(PermissionError,match="bound or automation coverage"):
        BundleRegistry._quality_gate(manifest,snapshot,evidence)


def test_gate_rejects_tiny_synthetic_cherry_picked_or_leaked_evidence():
    for modification in ("tiny","synthetic","omitted","duplicate","leak","calibration-leak"):
        manifest,snapshot,evidence=gate_inputs(10 if modification=="tiny" else 300)
        if modification=="synthetic":
            manifest["synthetic"]=True
        elif modification=="omitted":
            evidence["rows"].pop()
        elif modification=="duplicate":
            evidence["rows"][-1]=evidence["rows"][0]
        elif modification=="leak":
            snapshot["train"][0]["group_id"]=snapshot["test"][0]["group_id"]
        elif modification=="calibration-leak":
            snapshot["calibration"][0]["group_id"]="different"
        with pytest.raises((PermissionError,ValueError)):
            BundleRegistry._quality_gate(manifest,snapshot,evidence)


def test_gate_expiration_is_required_and_bounded():
    for change in ({"valid_until":time.time()-1}, {"observed_at":time.time()+100},
                   {"valid_until":time.time()+31*86400}, {"observed_at":None}):
        manifest,snapshot,evidence=gate_inputs()
        with pytest.raises((ValueError,PermissionError)):
            BundleRegistry._quality_gate(manifest,snapshot,{**evidence,**change})


def test_active_and_rollback_reject_expired_selective_evidence(setup_registry):
    store,registry,model_dir,manifest=setup_registry
    registry.register(manifest,"workspace",model_dir)
    registry.promote("bundle-1","workspace",template_commitment="template",language="en")
    # Inject a previously accepted historical gate to isolate time enforcement
    # from the separate evidence-quality test.
    with store.transaction() as state:
        deployment = next(iter(state["deployments"].values()))
        deployment["mode"]="selective"
        deployment["gate"]={"valid_until":time.time()-1}
    with pytest.raises(PermissionError,match="expired"):
        registry.active("workspace","template","en",verify_artifacts=False)
    manifest["id"]="bundle-2"
    registry.register(manifest,"workspace",model_dir)
    registry.promote("bundle-2","workspace",template_commitment="template",language="en")
    with pytest.raises(PermissionError,match="expired"):
        registry.rollback("workspace","template","en")


def test_gate_policy_must_be_fixed_before_final_test():
    for change in ("absent","different","future-registration"):
        manifest,snapshot,evidence=gate_inputs()
        if change=="absent":
            del manifest["selective_policy"]
        elif change=="different":
            evidence["threshold"]=.9
        else:
            manifest["registered_at"]=evidence["observed_at"]+1
        with pytest.raises((ValueError,PermissionError)):
            BundleRegistry._quality_gate(manifest,snapshot,evidence)


def test_registry_rejects_floating_model_revision(setup_registry):
    _,registry,model_dir,manifest=setup_registry
    manifest["model_revision"]="current-main-branch"
    with pytest.raises(ValueError,match="full lowercase commit"):
        registry.register(manifest,"workspace",model_dir)


@pytest.mark.parametrize("digest", [None, "b" * 64])
def test_registry_rejects_calibration_for_unbound_or_different_checkpoint(setup_registry, digest):
    _, registry, model_dir, manifest = setup_registry
    weight_hash = manifest["files"]["weights.bin"]
    (model_dir / "weights.bin").rename(model_dir / "model.safetensors")
    manifest["files"] = {"model.safetensors": weight_hash}
    manifest["calibrations"] = [fit_temperature([{"yes":.9,"no":.1}], ["yes"],
        model_bundle_id=manifest["id"], template_commitment="template", question_id="q", language="en",
        example_ids=["calibration-group"], model_weights_sha256=digest)]
    with pytest.raises(ValueError, match="checkpoint digest"):
        registry.register(manifest, "workspace", model_dir)


def test_registry_accepts_calibration_only_with_exact_weights_and_live_lineage(setup_registry):
    store, registry, model_dir, manifest = setup_registry
    store.add_grant(workspace_id="workspace", rights=["ai_use", "private_training"],
                    expires_at=time.time()+3600, evidence="synthetic fixture")
    template = {"id":"rubric", "version":1, "language":"en", "questions":[{"id":"q",
        "labels":[{"id":"yes"}, {"id":"no"}]}]}
    for i in range(3):
        store.record_evaluation(evaluation_id=f"eval{i}",workspace_id="workspace",case_id=f"case{i}",
            input_commitment=f"input{i}",template_commitment="template",template=template,input_payload={"text":f"Example {i}"})
        store.add_feedback(workspace_id="workspace",evaluation_id=f"eval{i}",input_commitment=f"input{i}",
            template_commitment="template",annotator_id="synthetic",labels={"q":"yes"},exposed_to_ai=False,independent_human=True)
    snapshot=store.create_snapshot("workspace","rubric",1)
    store.register_model_lineage(manifest["id"],snapshot["id"],"workspace")
    weight_hash=manifest["files"]["weights.bin"]
    (model_dir / "weights.bin").rename(model_dir / "model.safetensors")
    manifest["files"]={"model.safetensors":weight_hash}
    manifest["snapshot_id"]=snapshot["id"]
    manifest["calibrations"]=[fit_temperature([{"yes":.9,"no":.1}], ["yes"],
        model_bundle_id=manifest["id"],template_commitment="template",question_id="q",language="en",
        example_ids=[snapshot["calibration"][0]["group_id"]],model_weights_sha256=weight_hash)]
    envelope=registry.register(manifest,"workspace",model_dir)
    assert envelope["manifest"]["calibrations"][0]["model_weights_sha256"] == weight_hash
