import json
import os
import socket

import pytest

from rateloop_evaluator.backends import (GLiClassBackend, GLICLASS_FILES,
    provision_gliclass_model, validate_gliclass_model, write_model_manifest)


def test_comparison_provisioning_rejects_floating_revision(tmp_path):
    with pytest.raises(ValueError, match="commit SHA"):
        provision_gliclass_model(tmp_path, revision="latest")


def test_comparison_rejects_nested_remote_encoder_code(tmp_path):
    for name in GLICLASS_FILES[:-1]:
        (tmp_path / name).write_text("{}" if name.endswith(".json") else "weights")
    (tmp_path / "config.json").write_text(json.dumps({"architecture_type": "uni-encoder",
        "encoder_config": {"model_type": "modernbert", "auto_map": {"AutoModel": "remote.Code"}}}))
    write_model_manifest(tmp_path, source={})
    with pytest.raises(ValueError, match="Remote custom"):
        validate_gliclass_model(tmp_path)


def test_comparison_labels_include_meaning_and_preserve_identity():
    question = {"id": "tone", "labels": [{"id": "yes", "description": "Polite"},
                                           {"id": "no", "description": "Rude"}]}
    assert GLiClassBackend._labels(question) == ["yes: Polite", "no: Rude"]


@pytest.mark.skipif(not os.environ.get("RATELOOP_TEST_GLICLASS_DIR"), reason="Separate comparison environment and checkpoint required")
def test_real_gliclass_uses_checkpoint_markers_and_all_scores_offline(monkeypatch):
    def reject(*args, **kwargs):
        raise AssertionError("GLiClass attempted network access")
    monkeypatch.setattr(socket.socket, "connect", reject)
    questions = [{"id": "tone", "text": "What is the sentiment?", "labels": [
        {"id": "positive", "description": "Happy and satisfied"},
        {"id": "negative", "description": "Unhappy or disappointed"}], "passLabels": ["positive"]}]
    backend = GLiClassBackend(os.environ["RATELOOP_TEST_GLICLASS_DIR"],
                               os.environ.get("RATELOOP_TEST_DEVICE", "cpu"))
    scores = backend.predict("Thank you, I love this product!", questions)
    assert set(scores["tone"]) == {"positive", "negative"}
    assert sum(scores["tone"].values()) == pytest.approx(1.0)
    pipe = backend.pipeline.pipe
    assert pipe.tokenizer.convert_tokens_to_ids(pipe.label_token) == backend._model.config.class_token_index
    assert pipe.tokenizer.convert_tokens_to_ids(pipe.sep_token) == backend._model.config.text_token_index
