import json
from pathlib import Path
import unicodedata

from rateloop_evaluator.learning import _digest, _source_aliases
from rateloop_evaluator.protocol import commitment
from rateloop_evaluator.text_normalization import normalize_learning_text, _DATA

FIXTURE = json.loads((Path(__file__).parents[1] / "contracts/learning-input-normalization.v1.json").read_text())


def test_shared_fixture_matches_independent_unicode_oracle():
    assert _DATA["unicodeVersion"] == FIXTURE["unicodeVersion"] == "14.0.0"
    # Python's native full casefold is an independent oracle for all mappings,
    # not another implementation consuming our frozen table.
    folds = {chr(cp): chr(cp).casefold() for cp in range(0x110000) if chr(cp).casefold() != chr(cp)}
    assert folds == _DATA["caseFolds"]
    assert len(_DATA["caseFolds"]) == 1530
    assert _DATA["whitespaceCodePoints"] == [cp for cp in range(0x110000) if chr(cp).isspace()]
    probe = FIXTURE["exhaustiveProbe"]
    assert probe["scalarCount"] == 6278
    assert probe["expected"] == " ".join(unicodedata.normalize("NFKC", probe["input"]).casefold().split())
    for case in [*FIXTURE["cases"], probe]:
        assert normalize_learning_text(case["input"]) == case["expected"]


def test_worker_aliases_use_canonical_contract_and_retain_historical_exclusions():
    for case in FIXTURE["cases"]:
        input_value = {"text": case["input"], "context": "", "evidence": ""}
        row = {"case_id": "case", "group_id": "group", "input": input_value}
        aliases = _source_aliases(row)
        expected = {**input_value, "text": case["expected"]}
        assert aliases[-1] == "normalized_input:" + _digest(expected)
        legacy = {key: " ".join(unicodedata.normalize("NFKC", value).casefold().split()) for key, value in input_value.items()}
        assert "normalized_input:" + _digest(legacy) in aliases
        assert commitment(expected, "rateloop.evaluator.learning-input.v1") == case["commitment"]
    assert normalize_learning_text("\U0001e030") == "\U0001e030", "Unicode15 additions stay opaque"
    assert normalize_learning_text("\U0001ccd6") == "\U0001ccd6", "Unicode16 additions stay opaque"
