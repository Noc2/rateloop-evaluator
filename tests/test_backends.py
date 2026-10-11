import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

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


def test_decide_requires_explicit_selection_and_preserves_separate_pinned_identity(tmp_path, monkeypatch):
    from rateloop_evaluator.backends import MODEL_ID, DECIDE_MODEL_ID, DECIDE_MODEL_REVISION
    calls = []
    def download(repository, *, revision, local_dir, allow_patterns, token):
        calls.append((repository, revision, token))
        root = Path(local_dir)
        for relative in MODEL_FILES:
            path = root / relative; path.parent.mkdir(exist_ok=True)
            path.write_text("---\nlicense: apache-2.0\n---\n" if relative == "README.md" else
                json.dumps({"architecture": "boundary"}) if relative.endswith(".json") else "weights")
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    base = provision_model(tmp_path / "base")
    candidate = provision_model(tmp_path / "decide", checkpoint="decide")
    assert calls == [(MODEL_ID, MODEL_REVISION, False), (DECIDE_MODEL_ID, DECIDE_MODEL_REVISION, False)]
    assert base["source"]["repository"] != candidate["source"]["repository"]
    from rateloop_evaluator.training import assert_public_training_base
    with pytest.raises(PermissionError, match="reviewed public"):
        assert_public_training_base(candidate)
    with pytest.raises(ValueError, match="reviewed pinned revision"):
        provision_model(tmp_path / "wrong-revision", checkpoint="decide", revision="0" * 40)
    with pytest.raises(ValueError, match="base or decide"):
        provision_model(tmp_path / "unknown", checkpoint="unknown")


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



def test_checkpoint_cache_hint_is_read_only_scoped_and_never_follows_symlink(tmp_path,monkeypatch):
    from rateloop_evaluator import backends
    weights=tmp_path/'model.safetensors';weights.write_bytes(b'Immutable test weights')
    calls=[]
    monkeypatch.setattr(backends.os,'POSIX_FADV_DONTNEED',4,raising=False)
    monkeypatch.setattr(backends.os,'posix_fadvise',lambda descriptor,offset,length,strategy:
        calls.append((backends.os.read(descriptor,100),offset,length,strategy)),raising=False)
    backends.release_checkpoint_file_cache(tmp_path)
    assert calls==[(b'Immutable test weights',0,0,4)] and weights.read_bytes()==b'Immutable test weights'
    weights.unlink();weights.symlink_to(tmp_path/'other')
    (tmp_path/'other').write_bytes(b'Unrelated private file')
    backends.release_checkpoint_file_cache(tmp_path)
    assert len(calls)==1


def test_unsupported_checkpoint_cache_hint_does_not_change_inference_contract(tmp_path,monkeypatch):
    from rateloop_evaluator import backends
    monkeypatch.setattr(backends.os,'posix_fadvise',None,raising=False)
    backends.release_checkpoint_file_cache(tmp_path)


def test_semantic_labels_bind_question_answer_training_examples_and_decoding(monkeypatch):
    from copy import deepcopy
    from rateloop_evaluator.backends import model_label_tokens, question_schema, question_examples, score_adapter_version
    from rateloop_evaluator.training import training_records
    question = deepcopy(QUESTIONS[0])
    question['labels'] = [{'id':'approved','description':'Yes'}, {'id':'rejected','description':'No'}]
    question['examples'] = [{'text':'A synthetic demonstration','labelId':'approved'}]
    question['passLabels'] = ['approved']
    tokens = model_label_tokens(question, 2)
    assert tokens['approved'] == 'Is the reply polite? Answer: Yes'
    assert tokens['rejected'] == 'Is the reply polite? Answer: No'
    assert model_label_tokens(question, 1) == {'approved':'approved','rejected':'rejected'}
    class Schema:
        def classification(self, task, labels, **kwargs): self.task, self.labels, self.options = task, labels, kwargs
    monkeypatch.setitem(sys.modules, 'gliner2', SimpleNamespace(Schema=Schema))
    schema = question_schema([question], 2)
    record = training_records([{'input':{'text':'An independent training reply'},'template':{'questions':[question]},'labels':{'tone':'approved'}}], 2)[0]
    classification = record['output']['classifications'][0]
    assert set(schema.labels) == set(classification['labels']) == set(tokens.values())
    assert classification['true_label'] == [tokens['approved']]
    assert schema.options['examples'] == classification['examples'] == question_examples(question, 2)['examples']
    scores = validate_scores({'tone':[{'label':tokens['approved'],'confidence':.7},{'label':tokens['rejected'],'confidence':.3}]},[question],2)
    assert scores == {'tone':{'approved':.7,'rejected':.3}}
    changed = {**question,'text':'Is the reply factually correct?'}
    assert model_label_tokens(changed, 2) != tokens
    for version in (0, 3, True):
        with pytest.raises(ValueError, match='adapter'): score_adapter_version({'training':{'scoreAdapterVersion':version}})


@pytest.mark.skipif(not os.environ.get('RATELOOP_TEST_MODEL_DIR'), reason='Explicit pinned local checkpoint required')
def test_real_semantic_adapter_keeps_transport_labels_and_complete_scores():
    backend = GLiNERBackend(os.environ['RATELOOP_TEST_MODEL_DIR'], os.environ.get('RATELOOP_TEST_DEVICE','cpu'), adapter_version=2)
    scores = backend.predict('Thank you for your patience. I will help you now.', QUESTIONS)
    assert set(scores['tone']) == {'yes','no'}
    assert sum(scores['tone'].values()) == pytest.approx(1., abs=1e-4)
    assert backend.adapter_version == 2
    backend.unload()
