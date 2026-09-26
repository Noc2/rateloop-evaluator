"""User-owned upload -> immutable snapshot -> comparison, without fabricated reviews."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rateloop_evaluator import cli
from rateloop_evaluator.learning import is_independent_reference
from rateloop_evaluator.protocol import commitment
from rateloop_evaluator.registry import BundleRegistry
from rateloop_evaluator.templates import custom_text_seed
from test_cli import initialized, invoke, model_files
from test_registry import gate_inputs


def uploaded(initialized,tmp_path,capsys,provenance="owner"):
    template=custom_text_seed("en")
    template_file=tmp_path/"template.json"; template_file.write_text(template.model_dump_json())
    upload=tmp_path/"examples.jsonl"
    upload.write_text("\n".join(json.dumps({"case_id":f"case-{i}","group_id":f"source-{i}",
        "text":f"User example {i}","label":"approved" if i%2 else "rejected"}) for i in range(12)))
    preview_file=tmp_path/"preview.json"
    report=invoke(capsys,initialized,"dataset-preview","--file",upload,"--template-file",template_file,
        "--format","jsonl","--output",preview_file)
    assert report["row_count"]==12 and "User example" not in json.dumps(report)
    assert json.loads(preview_file.read_text())["rows"][0]["input"]["text"]=="User example 0"
    command=("dataset-import","--file",upload,"--template-file",template_file,"--format","jsonl",
        "--dataset-id","user-examples","--provenance",provenance,"--evidence","Owner's synthetic integration fixture")
    invoke(capsys,initialized,*command,expected=1)
    invoke(capsys,initialized,"grant","--right","private_training","--template",template.id,
        "--field","input.text","--field","imported_labels","--evidence","Owner permits private dataset learning")
    imported=invoke(capsys,initialized,*command)
    assert not imported["independent_reference"]
    snap=invoke(capsys,initialized,"snapshot","--template",template.id,"--version","1",
        "--template-commitment",commitment(template.model_dump(),"rateloop.evaluator.template.v1"),
        "--dataset-version",imported["id"],"--no-feedback")
    _,_,store,_=cli.state(SimpleNamespace(state_dir=initialized))
    with store.transaction() as state:
        assert not state["evaluations"] and not state["feedback"] and not state["deployments"]
        assert not any("ai_use" in grant["rights"] for grant in state["grants"].values())
    return template,store,snap


def test_cli_user_dataset_comparison_does_not_activate_or_claim_independent_quality(initialized,tmp_path,capsys,monkeypatch):
    template,store,snap=uploaded(initialized,tmp_path,capsys)
    from rateloop_evaluator import backends
    class LocalModel:
        def __init__(self,path,device): self.manifest={"source":{"model":str(path)}}
        def count_tokens(self,*_): return 20
        def predict(self,text,questions): return {"judgment":{"approved":.8,"rejected":.2}}
    monkeypatch.setattr(backends,"GLiNERBackend",LocalModel)
    output=tmp_path/"comparison.json"
    report=invoke(capsys,initialized,"compare","--snapshot-id",snap["snapshotId"],
        "--model",f"base={tmp_path}/base","--model",f"candidate={tmp_path}/candidate","--output",output)
    assert report["qualityClaim"] is False and report["activationChanged"] is False
    comparison=json.loads(output.read_text())
    assert comparison["independent_reference_count"]==0
    assert comparison["provenance_counts"]=={"owner":comparison["test_group_count"]}
    assert comparison["models"]["base"]["agreement"]==comparison["models"]["candidate"]["agreement"]
    assert comparison["quality_gate"] is False
    with store.transaction() as state: assert not state["deployments"]


@pytest.mark.parametrize("provenance",["owner","ai_assisted","synthetic"])
def test_imports_cannot_become_independent_calibration_or_qualification(initialized,tmp_path,capsys,monkeypatch,provenance):
    _,store,snap=uploaded(initialized,tmp_path,capsys,provenance)
    model=model_files(tmp_path,training={"bundleId":"candidate","workspaceId":"workspace-test","snapshotId":snap["snapshotId"]})
    from rateloop_evaluator import backends
    monkeypatch.setattr(backends,"GLiNERBackend",lambda *_:pytest.fail("Qualification attempted model inference on uploaded reference labels"))
    failure=invoke(capsys,initialized,"calibrate","--snapshot-id",snap["snapshotId"],"--model-dir",model,
        "--bundle-id","candidate","--output",tmp_path/"calibration.json",expected=1)
    assert "independently collected blind human references" in failure
    assert not (tmp_path/"calibration.json").exists()
    original_state=cli.state
    def state(args):
        root,local,store,_=original_state(args)
        registry=SimpleNamespace(get=lambda *_:{"manifest":{"snapshot_id":snap["snapshotId"],"selective_policy":{"threshold":.9}}})
        return root,local,store,registry
    monkeypatch.setattr(cli,"state",state)
    failure=invoke(capsys,initialized,"score-test","--bundle-id","candidate","--output",tmp_path/"evidence.json",expected=1)
    assert "independently collected blind human references" in failure
    assert not (tmp_path/"evidence.json").exists()
    manifest,snapshot,evidence=gate_inputs()
    snapshot["test"][0].update(source_kind="dataset",independent_reference=True,label_provenance=provenance)
    assert not is_independent_reference(snapshot["test"][0])
    with pytest.raises(PermissionError,match="independently collected blind human references"):
        BundleRegistry._quality_gate(manifest,snapshot,evidence)
