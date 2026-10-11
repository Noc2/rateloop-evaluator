"""Paired final-sample diagnostics; these do not authorize model activation."""
import math


def paired_correctness(baseline: list[bool], candidate: list[bool]) -> dict:
    if not baseline or len(baseline) != len(candidate) or any(type(value) is not bool for value in baseline+candidate):
        raise ValueError('Paired diagnostics require aligned Boolean correctness observations')
    wins = sum(b is False and c is True for b,c in zip(baseline,candidate))
    losses = sum(b is True and c is False for b,c in zip(baseline,candidate))
    discordant = wins+losses
    # Exact one-sided McNemar conditional test under equally likely directions.
    # Abstention/overflow is incorrect and remains in the full source denominator.
    if not discordant or not wins:
        probability = 1.
    else:
        terms = [math.lgamma(discordant+1)-math.lgamma(k+1)-math.lgamma(discordant-k+1)
                 - discordant*math.log(2) for k in range(wins, discordant+1)]
        offset = max(terms)
        probability = min(1., math.exp(offset)*sum(math.exp(value-offset) for value in terms))
    return {'sample_count': len(baseline), 'candidate_only_correct': wins,
        'baseline_only_correct': losses, 'both_correct': sum(b and c for b,c in zip(baseline,candidate)),
        'both_incorrect': sum(not b and not c for b,c in zip(baseline,candidate)),
        'agreement_difference': (wins-losses)/len(baseline), 'method': 'exact_one_sided_mcnemar',
        'p_value': probability, 'quality_gate': False}


def holm_adjust(reports: list[dict]) -> None:
    """Adjust the complete declared diagnostic family, never select comparisons."""
    previous = 0.
    for rank, report in enumerate(sorted(reports, key=lambda row: row['p_value'])):
        previous = max(previous, min(1., (len(reports)-rank)*report['p_value']))
        report['holm_adjusted_p_value'] = previous
