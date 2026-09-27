"""Small authored bilingual regression cases, never qualification evidence."""
from __future__ import annotations

from datetime import datetime, timezone
import math
import time

from .backends import render_input
from .benchmark import percentile, synchronize
from .comparison import label_metrics
from .protocol import commitment
from .templates import custom_text_evaluation


def diagnostic_cases():
    """24 synthetic cases with labels fixed before model execution.

    These examples exercise contradictions, missing evidence, irrelevant numbers,
    and presence rather than absence claims. They are public regression fixtures,
    not representative data or a training/held-out independent benchmark.
    """
    cases = []
    tasks = {
        'en': [
            ('criterion', 'Does the text state an explicit numeric budget?', [
                ('budget', 'The exhibition has a budget of EUR 450.', '', True),
                ('unrelated-number', 'The exhibition has 24 paintings; the budget is undecided.', '', False),
                ('negation', 'No budget amount has been set; 12 people will attend.', '', False),
                ('zero', 'The project budget is EUR 0.', '', True),
            ]),
            ('evidence', 'Is every factual claim in the text supported by the supplied evidence?', [
                ('supported', 'The archive opens at 09:00.', 'The archive opens daily at 09:00.', True),
                ('contradiction', 'The archive opens at 10:00.', 'The archive opens daily at 09:00.', False),
                ('missing', 'The archive opens at 09:00.', 'The archive has a blue entrance.', False),
                ('partial', 'The archive opens at 09:00 and entry is free.', 'The archive opens daily at 09:00.', False),
            ]),
            ('pii', 'Does the text contain an email address?', [
                ('present', 'Contact alex@example.test for details.', '', True),
                ('placeholder', 'Contact [EMAIL] for details.', '', False),
                ('negated-claim', 'There is no email address here: alex@example.test.', '', True),
                ('url', 'Read https://example.test/contact for details.', '', False),
            ]),
        ],
        'de': [
            ('criterion', 'Nennt der Text ein konkretes numerisches Budget?', [
                ('budget', 'Die Ausstellung hat ein Budget von 450 Euro.', '', True),
                ('unrelated-number', 'Die Ausstellung hat 24 Gemälde; das Budget steht noch nicht fest.', '', False),
                ('negation', 'Es wurde kein Budgetbetrag festgelegt; 12 Personen nehmen teil.', '', False),
                ('zero', 'Das Projektbudget beträgt 0 Euro.', '', True),
            ]),
            ('evidence', 'Wird jede Tatsachenbehauptung im Text durch die bereitgestellten Belege gestützt?', [
                ('supported', 'Das Archiv öffnet um 09:00 Uhr.', 'Das Archiv öffnet täglich um 09:00 Uhr.', True),
                ('contradiction', 'Das Archiv öffnet um 10:00 Uhr.', 'Das Archiv öffnet täglich um 09:00 Uhr.', False),
                ('missing', 'Das Archiv öffnet um 09:00 Uhr.', 'Das Archiv hat einen blauen Eingang.', False),
                ('partial', 'Das Archiv öffnet um 09:00 Uhr und der Eintritt ist kostenlos.', 'Das Archiv öffnet täglich um 09:00 Uhr.', False),
            ]),
            ('pii', 'Enthält der Text eine E-Mail-Adresse?', [
                ('present', 'Weitere Informationen erhalten Sie unter alex@example.test.', '', True),
                ('placeholder', 'Weitere Informationen erhalten Sie unter [EMAIL].', '', False),
                ('negated-claim', 'Hier steht keine E-Mail-Adresse: alex@example.test.', '', True),
                ('url', 'Weitere Informationen stehen auf https://example.test/kontakt.', '', False),
            ]),
        ],
    }
    for language, entries in tasks.items():
        positive, negative = ('Yes', 'No') if language == 'en' else ('Ja', 'Nein')
        for task, prompt, examples in entries:
            template = custom_text_evaluation(language, prompt, positive, negative).model_dump()
            for edge, text, evidence, expected in examples:
                cases.append({'id': f'{task}-{language}-{edge}', 'language': language, 'task': task,
                              'edge': edge, 'template': template,
                              'input': {'text': text, 'context': '', 'evidence': evidence},
                              'expected': 'approved' if expected else 'rejected'})
    return cases


def _summary(rows):
    labels = ('approved', 'rejected')
    confusion = {label: {prediction: 0 for prediction in labels} for label in labels}
    expected = {label: sum(row['expected'] == label for row in rows) for label in labels}
    for row in rows:
        if row['predicted'] is not None:
            confusion[row['expected']][row['predicted']] += 1
    return {'count': len(rows), 'correct': sum(row['correct'] for row in rows),
            'abstentions': sum(row['predicted'] is None for row in rows),
            'confusion': confusion, 'expected_label_counts': expected,
            **label_metrics(confusion, expected),
            'prediction_latency_ms': {'p50': percentile([row['prediction_ms'] for row in rows], .5),
                                      'p95': percentile([row['prediction_ms'] for row in rows], .95)}}


def run_diagnostics(backend):
    cases, results = diagnostic_cases(), []
    device = getattr(backend, 'device', 'cpu')
    for case in cases:
        questions, text = case['template']['questions'], render_input(case['input'])
        token_count = backend.count_tokens(text, questions)
        if token_count > case['template']['maxTokens']:
            raise ValueError('Diagnostic case exceeds its token budget; no truncation is allowed')
        synchronize(device)
        started = time.perf_counter()
        scores = backend.predict(text, questions)
        synchronize(device)
        duration = (time.perf_counter() - started) * 1000
        if not isinstance(scores, dict) or set(scores) != {'judgment'} or not isinstance(scores['judgment'], dict):
            raise ValueError('Diagnostic prediction does not cover its exact criterion')
        distribution = scores['judgment']
        if (set(distribution) != {'approved', 'rejected'}
                or any(type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1
                       for score in distribution.values())):
            raise ValueError('Diagnostic prediction contains invalid raw scores')
        winners = [label for label, score in distribution.items() if score == max(distribution.values())]
        predicted = winners[0] if len(winners) == 1 else None
        results.append({key: case[key] for key in ('id', 'language', 'task', 'edge', 'expected')} |
                       {'predicted': predicted, 'correct': predicted == case['expected'],
                        'raw_scores': distribution, 'token_count': token_count, 'prediction_ms': duration})
    slices = {}
    for row in results:
        key = row['task'] + '/' + row['language']
        slices.setdefault(key, []).append(row)
    manifest = getattr(backend, 'manifest', None) or {}
    return {'schema_version': 'rateloop.synthetic-diagnostics.v1',
            'observed_at': datetime.now(timezone.utc).isoformat(),
            'suite_commitment': commitment(cases, 'rateloop.synthetic-diagnostics.v1'),
            'model_manifest_commitment': commitment(manifest, 'rateloop.diagnostic-model.v1'),
            'model': {key: value for key, value in manifest.get('source', {}).items()
                      if key in ('repository', 'revision', 'license')},
            'device': device, 'case_count': len(cases), 'label_provenance': 'synthetic',
            'independent_reference_count': 0, 'quality_gate': False, 'activation_changed': False,
            'overall': _summary(results), 'slices': {key: _summary(rows) for key, rows in slices.items()},
            'error_ids': [row['id'] for row in results if not row['correct']], 'cases': results,
            'limits': 'Authored public regression cases, not representative accuracy or calibration evidence. Scores are raw. Latency measures sequential prediction after tokenization; excludes loading, network and queues. No training, registration or activation occurs.'}
