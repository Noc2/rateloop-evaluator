"""Bind portable wording, inference, training and identity to the same examples."""
from copy import deepcopy
import json
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

import pytest

from rateloop_evaluator.backends import GLiNERBackend, question_schema
from rateloop_evaluator.protocol import Question, Template, commitment, validate_no_demonstration_overlap
from rateloop_evaluator.templates import custom_text_evaluation, is_custom_text_template, website_binary_question
from rateloop_evaluator.training import training_records


def question():
    return custom_text_evaluation("en", "Is a budget stated?", "Yes", "No").model_dump()["questions"][0]


def test_typescript_demonstration_fixture_has_identical_python_commitment():
    fixture = json.loads((Path(__file__).parent / "fixtures/evaluator-examples-v1.json").read_text())
    template = Template.model_validate(fixture["template"])
    assert template.model_dump() == fixture["template"]
    assert commitment(template.model_dump(), "rateloop.evaluator.template.v1") == fixture["templateCommitment"]


def sample(q):
    return {"input": {"text": "The budget is EUR 500."}, "template": {"questions": [q]},
            "labels": {"judgment": "approved"}}


@pytest.fixture
def schema_stub(monkeypatch):
    class Schema:
        def __init__(self): self.schema = {"classifications": []}
        def classification(self, task, labels, **kwargs):
            self.schema["classifications"].append({"task": task, "labels": labels, **kwargs})
    monkeypatch.setitem(sys.modules, "gliner2", SimpleNamespace(Schema=Schema))


def test_omitted_and_empty_demonstrations_preserve_existing_commitments(schema_stub):
    fixture = Path(__file__).parent / "fixtures/custom-text-templates.json"
    for item in json.loads(fixture.read_text()):
        original = Template.model_validate(item["template"])
        with_empty = deepcopy(item["template"])
        with_empty["questions"][0]["examples"] = []
        assert Template.model_validate(with_empty).model_dump() == original.model_dump()
        assert commitment(original.model_dump(), "rateloop.evaluator.template.v1") == item["templateCommitment"]
    q = question()
    assert "examples" not in question_schema([q]).schema["classifications"][0]
    assert "examples" not in training_records([sample(q)])[0]["output"]["classifications"][0]


def test_normalized_demonstrations_bind_all_consumers_and_template_identity(schema_stub):
    q = question()
    q["examples"] = [{"text": " \ufeffFunding is EUR 500.\t ", "labelId": "approved"},
                     {"text": "We need funding.", "labelId": "rejected"}]
    expected = [["Funding is EUR 500.", "approved"], ["We need funding.", "rejected"]]
    assert question_schema([q]).schema["classifications"][0]["examples"] == expected
    assert training_records([sample(q)])[0]["output"]["classifications"][0]["examples"] == expected
    template = custom_text_evaluation("en", q["text"], "Yes", "No", q["examples"])
    assert is_custom_text_template(template)
    assert website_binary_question(template).examples[0].text == expected[0][0]
    base = custom_text_evaluation("en", q["text"], "Yes", "No")
    digest = lambda value: commitment(value.model_dump(), "rateloop.evaluator.template.v1")
    assert digest(template) != digest(base)
    changed = template.model_copy(deep=True)
    changed.questions[0].examples[0].labelId = "rejected"
    assert digest(changed) != digest(template)


@pytest.mark.parametrize("examples", [
    [{"text": " ", "labelId": "approved"}],
    [{"text": "Budget", "labelId": "missing"}],
    [{"text": "Budget", "labelId": "approved", "extra": True}],
    [{"text": "x", "labelId": "approved"}] * 5,
    [{"text": "x" * 601, "labelId": "approved"}],
    [{"text": "😀" * 301, "labelId": "approved"}],
    [{"text": "x" * 401, "labelId": "approved"}] * 4,
    [{"text": "Bad\u202etext", "labelId": "approved"}],
    [{"text": "Bad\x00text", "labelId": "approved"}],
    [{"text": "Bad\ud800text", "labelId": "approved"}],
    [{"text": 123, "labelId": "approved"}],
    None,
])
def test_invalid_demonstrations_fail_consistently_in_protocol_inference_and_training(schema_stub, examples):
    q = question(); q["examples"] = examples
    for consumer in (lambda: Question.model_validate(q), lambda: question_schema([q]),
                     lambda: training_records([sample(q)])):
        with pytest.raises(ValueError): consumer()


def test_utf16_and_total_boundary_with_multiline_and_whitespace():
    q = question()
    q["examples"] = [{"text": " " * 20 + "😀" * 300 + " " * 20, "labelId": "approved"},
                     {"text": "x" * 600, "labelId": "rejected"},
                     {"text": "x" * 398 + "\nY", "labelId": "rejected"}]
    assert len(Question.model_validate(q).examples) == 3


@pytest.mark.parametrize("text", ["Funding is EUR 500.", " Funding\tis EUR\n500. ",
                                "\ufeffFunding\u00a0is\u2028EUR 500.\ufeff"])
def test_disclosed_examples_cannot_be_training_or_holdout_evidence(text):
    q = question(); q["examples"] = [{"text": "Funding is EUR 500.", "labelId": "approved"}]
    with pytest.raises(ValueError, match="repeat a rubric"):
        validate_no_demonstration_overlap(text, [q])
    value = sample(q); value["input"]["text"] = text
    with pytest.raises(ValueError, match="repeat a rubric"):
        training_records([value])
    validate_no_demonstration_overlap("No funding is stated.", [q])


@pytest.mark.skipif(not os.environ.get("RATELOOP_TEST_MODEL_DIR"), reason="Explicit provisioned model required")
def test_real_pinned_processor_counts_and_uses_examples_without_network(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", lambda *_: (_ for _ in ()).throw(AssertionError("Network access")))
    backend = GLiNERBackend(os.environ["RATELOOP_TEST_MODEL_DIR"], os.environ.get("RATELOOP_TEST_DEVICE", "cpu"))
    q = question()
    text = "The project lists 24 milestones and no spending limit."
    base_count = backend.count_tokens(text, [q])
    q["examples"] = [{"text": "The project has 12 phases.", "labelId": "rejected"},
                     {"text": "The spending limit is EUR 800.", "labelId": "approved"}]
    count = backend.count_tokens(text, [q])
    assert count > base_count
    output = backend.predict(text, [q])["judgment"]
    assert set(output) == {"approved", "rejected"}
    assert abs(sum(output.values()) - 1) < 1e-4
    # Both upstream consumers receive the same complete sequence before optimizer augmentation.
    inference = question_schema([q]).schema["classifications"][0]
    train = training_records([sample(q)])[0]["output"]["classifications"][0]
    assert inference["examples"] == train["examples"]
    backend.unload()
