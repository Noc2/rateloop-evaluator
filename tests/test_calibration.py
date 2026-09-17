import math
from copy import deepcopy
import pytest
from rateloop_evaluator.calibration import apply_temperature, fit_temperature, false_approval_upper_bound, validate_calibration


def fit(**kwargs):
    return fit_temperature([{"yes":.99,"no":.01}]*4,["yes","no","yes","no"],model_bundle_id="bundle",
        template_commitment="template",question_id="q",language="en",example_ids=["a","b","c","d"],**kwargs)


def test_temperature_fit_reduces_overconfidence_and_binds_full_scope():
    artifact = fit()
    assert artifact["nll_after"] < artifact["nll_before"]
    args = dict(model_bundle_id="bundle",template_commitment="template",question_id="q",language="en")
    probs = apply_temperature({"yes":.99,"no":.01},artifact,**args)
    assert .5 < probs["yes"] < .7
    assert sum(probs.values()) == pytest.approx(1)
    for key in args:
        with pytest.raises(ValueError,match="mismatch"):
            apply_temperature({"yes":.99,"no":.01},artifact,**{**args,key:"other"})
    with pytest.raises(ValueError,match="label set"):
        apply_temperature({"a":.5,"b":.5},artifact,**args)


def test_invalid_calibration_and_scores_fail_closed():
    artifact=fit()
    artifact["temperature"] = .01
    with pytest.raises(ValueError):
        validate_calibration(artifact)
    for value in (float("nan"),float("inf"),-1,True):
        with pytest.raises(ValueError):
            fit_temperature([{"yes":value,"no":.1}],["yes"],model_bundle_id="b",template_commitment="t",
                            question_id="q",language="en",example_ids=["a"])
    with pytest.raises(ValueError,match="unique"):
        fit_temperature([{"yes":.9,"no":.1}]*2,["yes"]*2,model_bundle_id="b",template_commitment="t",
                        question_id="q",language="en",example_ids=["a","a"])


def test_exact_one_sided_bound_has_correct_zero_error_and_nonzero_cases():
    assert false_approval_upper_bound(0,300) == pytest.approx(1-.05**(1/300))
    assert false_approval_upper_bound(0,300) < .01
    assert false_approval_upper_bound(0,20) > .13
    # P[X <= 1] with n=2 is 1-p^2, giving a closed-form reference.
    assert false_approval_upper_bound(1,2) == pytest.approx(math.sqrt(.95))
    assert false_approval_upper_bound(1,300) > false_approval_upper_bound(0,300)
    assert false_approval_upper_bound(300,300) == 1
    for errors,total in ((0,0),(-1,10),(11,10),(True,10)):
        with pytest.raises(ValueError):
            false_approval_upper_bound(errors,total)
