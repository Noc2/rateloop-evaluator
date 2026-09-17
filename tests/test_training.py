import pytest
from rateloop_evaluator.training import TrainOptions, train_snapshot, training_records
from rateloop_evaluator.protocol import CaseInput


def example():
    return {"input": {"text": "Thank you", "context": "Support reply", "evidence": ""},
            "template": {"questions": [{"id": "tone", "text": "Is this polite?", "labels": [
                {"id": "yes", "description": "Courteous"}, {"id": "no", "description": "Rude"}]}]},
            "labels": {"tone": "yes"}}


def test_training_preserves_rubric_question_and_label_definitions():
    sample = example()
    record = training_records([sample])[0]
    assert record["input"] == CaseInput.model_validate(sample["input"]).render()
    task = record["output"]["classifications"][0]
    assert task["labels"] == ["yes", "no"]
    assert task["true_label"] == ["yes"]
    assert task["label_descriptions"] == {"yes": "Courteous", "no": "Rude"}
    assert task["prompt"] == "Is this polite?"
    assert task["multi_label"] is False


@pytest.mark.parametrize("labels", [{}, {"tone": "unknown"}, {"tone": "yes", "extra": "yes"}])
def test_training_rejects_incomplete_or_changed_human_labels(labels):
    sample = example()
    sample["labels"] = labels
    with pytest.raises(ValueError):
        training_records([sample])


def test_revocation_checked_before_any_model_load_or_output(tmp_path):
    class RevokedStore:
        def load_snapshot(self, snapshot_id, workspace_id):
            raise PermissionError("grant revoked")
    output = tmp_path / "output"
    with pytest.raises(PermissionError, match="revoked"):
        train_snapshot(RevokedStore(), "snapshot", "workspace", tmp_path / "missing", output, bundle_id="model")
    assert not output.exists()


@pytest.mark.parametrize("options", [TrainOptions(method="unknown"), TrainOptions(max_steps=0),
                                      TrainOptions(learning_rate=float("nan")), TrainOptions(epochs=0)])
def test_invalid_training_options_are_rejected(options):
    with pytest.raises(ValueError):
        options.validate()


def test_training_reports_drop_content_and_encode_unevaluated_metric_as_null():
    from rateloop_evaluator.training import sanitized_training_metrics
    result = sanitized_training_metrics({"best_metric": float("inf"), "raw_input": "private content",
        "train_metrics_history": [{"loss": .5, "example": "private content"}]})
    assert result == {"best_metric": None, "train_metrics_history": [{"loss": .5}], "eval_metrics_history": []}


def test_retraining_requires_original_public_weights_not_private_ancestry():
    from copy import deepcopy
    from rateloop_evaluator.backends import MODEL_ID, MODEL_REVISION
    from rateloop_evaluator.training import assert_public_training_base, REVIEWED_BASE_WEIGHTS_SHA256
    public = {"source": {"repository": MODEL_ID, "revision": MODEL_REVISION},
              "files": {"model.safetensors": REVIEWED_BASE_WEIGHTS_SHA256}}
    assert_public_training_base(public)
    for change in ("adapted", "stripped-metadata", "different-source", "floating-revision"):
        candidate = deepcopy(public)
        if change == "adapted": candidate["training"] = {"workspaceId": "other-workspace"}
        if change == "stripped-metadata": candidate["files"]["model.safetensors"] = "b" * 64
        if change == "different-source": candidate["source"]["repository"] = "other/model"
        if change == "floating-revision": candidate["source"]["revision"] = "main"
        with pytest.raises(PermissionError, match="reviewed public"):
            assert_public_training_base(candidate)
