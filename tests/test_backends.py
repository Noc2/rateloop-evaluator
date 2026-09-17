import json
import os
from pathlib import Path

import pytest

from rateloop_evaluator.backends import (GLiNERBackend, MODEL_FILES, MODEL_REVISION,
                                       provision_model, render_input, validate_local_model,
                                       validate_scores, write_model_manifest)
from rateloop_evaluator.protocol import CaseInput


QUESTIONS = [{"id": "tone", "text": "Is the reply polite?", "labels": [
    {"id": "yes", "description": "Courteous"}, {"id": "no", "description": "Rude"}],
    "passLabels": ["yes"]}]


def local_artifacts(tmp_path):
    for relative in MODEL_FILES[:-1]:
        path = tmp_path / relative
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({"architecture": "boundary"}) if relative.endswith(".json") else "weights")
    write_model_manifest(tmp_path, source={"revision": MODEL_REVISION})
    return tmp_path


def test_artifact_integrity_and_required_offline_assets(tmp_path):
    local_artifacts(tmp_path)
    validate_local_model(tmp_path)
    (tmp_path / "model.safetensors").write_text("modified")
    with pytest.raises(ValueError, match="integrity"):
        validate_local_model(tmp_path)


def test_artifact_manifest_cannot_escape_directory(tmp_path):
    local_artifacts(tmp_path)
    path = tmp_path / "rateloop-model.json"
    data = json.loads(path.read_text())
    data["files"]["../secret"] = "0" * 64
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="external"):
        validate_local_model(tmp_path)


def test_loading_rejects_remote_code(tmp_path):
    local_artifacts(tmp_path)
    (tmp_path / "encoder_config/config.json").write_text('{"auto_map":{"AutoModel":"unreviewed.Code"}}')
    write_model_manifest(tmp_path, source={})
    with pytest.raises(ValueError, match="Remote custom"):
        validate_local_model(tmp_path)


def test_provisioning_rejects_floating_revision_before_network(tmp_path):
    with pytest.raises(ValueError, match="commit SHA"):
        provision_model(tmp_path, revision="main")


@pytest.mark.parametrize("rows", [
    [{"label": "yes", "confidence": .9}],
    [{"label": "yes", "confidence": .9}, {"label": "yes", "confidence": .1}],
    [{"label": "yes", "confidence": float("nan")}, {"label": "no", "confidence": .1}],
    [{"label": "yes", "confidence": .9}, {"label": "no", "confidence": .9}],
])
def test_partial_duplicate_nonfinite_or_wrong_activation_scores_fail_closed(rows):
    with pytest.raises(ValueError):
        validate_scores({"tone": rows}, QUESTIONS)


def test_complete_scores_keep_stable_label_ids():
    assert validate_scores({"tone": [{"label": "no", "confidence": .1},
                                     {"label": "yes", "confidence": .9}]}, QUESTIONS) == {
        "tone": {"no": .1, "yes": .9}}


def test_training_and_service_use_identical_input_rendering():
    value = {"text": "Reply", "context": "Request", "evidence": "Policy"}
    assert render_input(value) == CaseInput.model_validate(value).render()


def test_constructor_is_lazy_and_local_only():
    backend = GLiNERBackend("/definitely-missing-model")
    assert backend._model is None
    with pytest.raises(FileNotFoundError):
        backend.load()


@pytest.mark.skipif(not os.environ.get("RATELOOP_TEST_MODEL_DIR"), reason="Explicit provisioned model required")
def test_real_checkpoint_scores_without_network(monkeypatch):
    import socket
    def reject(*args, **kwargs):
        raise AssertionError("Inference attempted network access")
    monkeypatch.setattr(socket.socket, "connect", reject)
    backend = GLiNERBackend(os.environ["RATELOOP_TEST_MODEL_DIR"], os.environ.get("RATELOOP_TEST_DEVICE", "cpu"))
    output = backend.predict("Thank you for your patience. We will investigate your request.", QUESTIONS)
    assert set(output["tone"]) == {"yes", "no"}
    assert backend.count_tokens("Thank you.", QUESTIONS) > backend.count_tokens("", QUESTIONS)


def test_relative_position_bucket_is_not_absolute_context_limit():
    from types import SimpleNamespace
    from rateloop_evaluator.backends import model_token_limit
    config = SimpleNamespace(max_position_embeddings=512, position_biased_input=False)
    model = SimpleNamespace(config=SimpleNamespace(max_len=4096), encoder=SimpleNamespace(config=config))
    assert model_token_limit(model) == 4096
    config.position_biased_input = True
    assert model_token_limit(model) == 512


def test_model_rejects_undeclared_files_and_symlink_directories(tmp_path):
    local_artifacts(tmp_path)
    (tmp_path / "unreviewed.bin").write_text("extra loader input")
    with pytest.raises(ValueError, match="undeclared"):
        validate_local_model(tmp_path)
    (tmp_path / "unreviewed.bin").unlink()
    (tmp_path / "linked").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        validate_local_model(tmp_path)


def test_provisioned_model_license_must_match_reviewed_release(tmp_path):
    from rateloop_evaluator.backends import verify_checkpoint_license
    card = tmp_path / "README.md"
    card.write_text("---\nlicense: other\n---\nlicense: apache-2.0\n")
    with pytest.raises(ValueError, match="license differs"):
        verify_checkpoint_license(tmp_path)
    card.write_text("---\nlicense: apache-2.0\n---\n")
    verify_checkpoint_license(tmp_path)
