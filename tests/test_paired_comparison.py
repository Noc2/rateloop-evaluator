import pytest
from rateloop_evaluator.paired_comparison import paired_correctness, holm_adjust


def test_exact_discordant_pair_counts_and_one_sided_tail():
    # Five improvements without regressions: probability = 1/32.
    result = paired_correctness([False]*5+[True]*5, [True]*10)
    assert result['candidate_only_correct'] == 5 and result['baseline_only_correct'] == 0
    assert result['p_value'] == pytest.approx(1/32)
    assert result['sample_count'] == 10 and result['quality_gate'] is False
    assert paired_correctness([False]*10,[False]*10)['p_value'] == 1
    assert paired_correctness([True]*10,[False]*10)['p_value'] == 1
    assert paired_correctness([True,False],[False,True])['p_value'] == pytest.approx(.75)


def test_holm_uses_all_comparators_and_is_monotone():
    values = [{'p_value': .01}, {'p_value': .04}, {'p_value': .03}]
    holm_adjust(values)
    assert [value['holm_adjusted_p_value'] for value in values] == [.03,.06,.06]


@pytest.mark.parametrize('before,after', [([],[]),([True],[True,False]),([1],[True])])
def test_unaligned_or_non_boolean_pairs_fail_closed(before,after):
    with pytest.raises(ValueError): paired_correctness(before,after)
