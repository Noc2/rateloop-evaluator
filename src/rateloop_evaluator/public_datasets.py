"""Offline, bounded conversions of explicitly obtained public training samples.

Source hashes bind an operator's local export; they do not authenticate its origin.
No download, permission grant, split reassignment or training occurs here.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import unicodedata

from .datasets import MAX_BYTES, MAX_ROWS, _unique_object, preview_dataset
from .protocol import commitment
from .templates import custom_text_evaluation

MAX_SOURCE_BYTES = 64 * 1024 * 1024
MAX_SOURCE_ROWS = 50_000
MAX_LINE_BYTES = 1024 * 1024
SOURCES = {
    'helpsteer2': ('nvidia/HelpSteer2', 'NVIDIA', 'external_human', 'owner'),
    'helpsteer3-principle': ('nvidia/HelpSteer3', 'NVIDIA', 'ai_assisted', 'ai_assisted'),
    'openpii1m': ('ai4privacy/pii-masking-openpii-1m', 'Ai4Privacy / Ai Suisse SA', 'synthetic', 'synthetic'),
}
ENTITY_TYPES = frozenset('DATE GIVENNAME SURNAME EMAIL CITY TITLE TELEPHONENUM AGE STREET BUILDINGNUM ZIPCODE IDCARDNUM CREDITCARDNUMBER DRIVERLICENSENUM GENDER TAXNUM SEX SOCIALNUM PASSPORTNUM'.split())
SCORES = frozenset(('helpfulness', 'correctness', 'coherence'))


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _normalized(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def _text(row, key):
    value = row.get(key)
    if not isinstance(value, str) or not value.strip() or '\x00' in value or len(value) > 100_000:
        raise ValueError('Required source text is missing or invalid')
    return value


def _read_source(path, expected):
    if not re.fullmatch(r'[0-9a-f]{64}', expected):
        raise ValueError('Expected SHA256 must contain 64 lowercase hex characters')
    with Path(path).open('rb') as stream:
        content = stream.read(MAX_SOURCE_BYTES + 1)
    if not content or len(content) > MAX_SOURCE_BYTES:
        raise ValueError('Source JSONL must be nonempty and at most 64 MiB; export a bounded sample first')
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError('Source file SHA256 does not match the explicitly pinned export')
    rows = []
    for line in content.splitlines():
        if not line.strip():
            continue
        if len(line) > MAX_LINE_BYTES or len(rows) >= MAX_SOURCE_ROWS:
            raise ValueError('Source exceeds the 1 MiB line or 50000-row limit')
        try:
            row = json.loads(line, object_pairs_hook=_unique_object,
                             parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Invalid JSON number')))
        except (ValueError, UnicodeError):
            raise ValueError('Source must contain valid UTF-8 JSONL with unique object keys') from None
        if not isinstance(row, dict):
            raise ValueError('Each source row must be an object')
        rows.append(row)
    if not rows:
        raise ValueError('Source contains no examples')
    return rows


def _context(row):
    context = row.get('context')
    if not isinstance(context, list) or not context or len(context) > 200:
        raise ValueError('Principle source requires a conversation context')
    messages = []
    for message in context:
        if not isinstance(message, dict) or message.get('role') not in ('user', 'assistant', 'system'):
            raise ValueError('Principle context requires valid messages')
        messages.append(message['role'] + ': ' + _text(message, 'content'))
    text = '\n'.join(messages)
    if len(text) > 100_000:
        raise ValueError('Principle context exceeds the input limit')
    return text


def _pii(row, entity):
    text = _text(row, 'source_text')
    masked = _text(row, 'masked_text')
    spans = row.get('privacy_mask')
    if not isinstance(spans, list) or len(spans) > 1000:
        raise ValueError('OpenPII requires the full privacy_mask annotation list')
    labels, occupied = set(), set()
    for span in spans:
        if not isinstance(span, dict):
            raise ValueError('OpenPII annotation is invalid')
        start, end, label = span.get('start'), span.get('end'), span.get('label')
        if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text)
                or label not in ENTITY_TYPES or span.get('value') != text[start:end]
                or occupied.intersection(range(start, end))):
            raise ValueError('OpenPII annotation offsets, value or entity type are invalid')
        occupied.update(range(start, end)); labels.add(label)
    # Placeholder indices and literal numbers are not new independent sources.
    # Reconstruct masking from validated spans: do not trust supplied masked_text.
    rebuilt, offset = [], 0
    for span in sorted(spans, key=lambda value: value['start']):
        rebuilt.extend((text[offset:span['start']], '[' + span['label'] + ']'))
        offset = span['end']
    rebuilt.append(text[offset:])
    mask_family = re.sub(r'\d+', '<number>', _normalized(''.join(rebuilt)))
    declared_family = re.sub(r'\d+', '<number>', _normalized(re.sub(r'\[([A-Z]+)_\d+\]', r'[\1]', masked)))
    if mask_family != declared_family:
        raise ValueError('OpenPII masked_text does not match validated source annotations')
    # Absence refers to the selected annotated entity only, not absence of all PII.
    return text, mask_family, entity in labels


def prepare_public_dataset(*, file, dataset: str, revision: str, source_sha256: str,
                           language: str, source_split: str, max_rows: int = MAX_ROWS,
                           score: str | None = None, positive_min: int | None = None,
                           negative_max: int | None = None, principle: str | None = None,
                           entity: str | None = None) -> dict:
    """Return a reproducible import plus local-only exclusion/provenance evidence.

    Upstream validation/test exports are deliberately refused: the current
    importer creates its own frozen partitions and cannot preserve such holdouts.
    """
    if dataset not in SOURCES or language not in ('en', 'de'):
        raise ValueError('Choose a supported dataset and English or German')
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('Pin the source repository to a full 40-character commit revision')
    if source_split != 'train':
        raise ValueError('Only source training splits can be prepared; keep upstream validation and test data held out')
    if type(max_rows) is not int or not 1 <= max_rows <= MAX_ROWS:
        raise ValueError('Output row limit must be between 1 and 2000')
    if dataset == 'helpsteer2':
        if (language != 'en' or score not in SCORES or type(positive_min) is not int
                or type(negative_max) is not int or not 0 <= negative_max < positive_min <= 4
                or principle is not None or entity is not None):
            raise ValueError('HelpSteer2 requires English, helpfulness/correctness/coherence and explicit nonoverlapping 0–4 thresholds')
        prompt = f'Does the response merit a {score} score of at least {positive_min} on a 0–4 scale for the request in context?'
        positive, negative = 'Meets the score threshold', 'Below the score threshold'
        conversion = {'score': score, 'positive_min': positive_min, 'negative_max': negative_max}
    elif dataset == 'helpsteer3-principle':
        if not isinstance(principle, str) or not principle.strip() or any(value is not None for value in (score, positive_min, negative_max, entity)):
            raise ValueError('Choose one exact HelpSteer3 principle; mixed-criterion training is unsupported')
        prompt = (f'Does the response fulfil this principle: {principle}?' if language == 'en'
                  else f'Erfüllt die Antwort dieses Prinzip: {principle}?')
        positive, negative = ('Fulfils the principle', 'Does not fulfil the principle') if language == 'en' else ('Prinzip erfüllt', 'Prinzip nicht erfüllt')
        conversion = {'principle': principle}
    else:
        if entity not in ENTITY_TYPES or any(value is not None for value in (score, positive_min, negative_max, principle)):
            raise ValueError('OpenPII1M requires one supported entity type')
        prompt = (f'Does the text contain a value of the annotated personal-data type {entity}?' if language == 'en'
                  else f'Enthält der Text einen Wert des annotierten personenbezogenen Datentyps {entity}?')
        positive, negative = ('Entity present', 'Entity absent') if language == 'en' else ('Datentyp vorhanden', 'Datentyp nicht vorhanden')
        conversion = {'entity': entity, 'negative_policy': 'No span annotated with the selected entity; no generated negatives'}
    template = custom_text_evaluation(language, prompt, positive, negative)
    raw_rows = _read_source(file, source_sha256)
    candidates, excluded = [], []
    for raw in raw_rows:
        source_hash = _hash(raw)
        def exclude(reason):
            excluded.append({'source_row_sha256': source_hash, 'reason': reason, 'source': raw})
        try:
            if raw.get('split', 'train') != 'train':
                raise ValueError('Non-training source split')
            if dataset == 'helpsteer2':
                text, context = _text(raw, 'response'), _text(raw, 'prompt')
                value = raw.get(score)
                if type(value) is not int or not 0 <= value <= 4:
                    raise ValueError('Invalid source score')
                if negative_max < value < positive_min:
                    exclude('ambiguous_score'); continue
                approved, family = value >= positive_min, _normalized(context)
            elif dataset == 'helpsteer3-principle':
                if raw.get('language') != {'en': 'english', 'de': 'german'}[language]:
                    exclude('other_language'); continue
                if raw.get('principle') != principle:
                    exclude('other_principle'); continue
                text, context = _text(raw, 'response'), _context(raw)
                if raw.get('fulfilment') not in ('Yes', 'No'):
                    raise ValueError('Invalid principle fulfilment')
                approved, family = raw['fulfilment'] == 'Yes', _normalized(context)
            else:
                if raw.get('language') != language:
                    exclude('other_language'); continue
                text, family, approved = _pii(raw, entity)
                context = ''
            payload = {'text': text, 'context': context, 'evidence': ''}
            case_id = 'public_' + _hash([SOURCES[dataset][0], payload])
            group_id = 'source_' + _hash([SOURCES[dataset][0], family])
            # Browser imports use source_group_id; the local import contract uses
            # group_id. Both aliases must preserve the same grouping invariant.
            row = {'case_id': case_id, 'group_id': group_id, 'source_group_id': group_id, **payload,
                   'label': 'approved' if approved else 'rejected'}
            candidates.append((row, raw))
        except (ValueError, TypeError, KeyError):
            # Malformed labels must never be silently converted to negative labels.
            raise ValueError(f'Source row {len(candidates)+len(excluded)+1} does not match the selected dataset schema') from None
    if not candidates:
        raise ValueError('No examples match the selected criterion and language')
    seen, groups = {}, defaultdict(list)
    for row, raw in sorted(candidates, key=lambda item: (_hash(item[0]), _hash(item[1]))):
        prior = seen.get(row['case_id'])
        if prior:
            if prior['label'] != row['label']:
                raise ValueError('Identical source input has conflicting labels; review source annotations')
            excluded.append({'source_row_sha256': _hash(raw), 'reason': 'duplicate_input', 'source': raw})
            continue
        seen[row['case_id']] = row
        groups[row['group_id']].append((row, raw))
    selected, encoded, byte_count = [], [], 0
    for group in sorted(groups):
        items = sorted(groups[group], key=lambda item: item[0]['case_id'])
        lines = [(_json(row)+'\n').encode() for row, _ in items]
        if len(selected)+len(items) > max_rows or byte_count+sum(map(len, lines)) > MAX_BYTES:
            excluded.extend({'source_row_sha256': _hash(raw), 'reason': 'output_capacity', 'source': raw} for _, raw in items)
            continue
        selected.extend(row for row, _ in items); encoded.extend(lines); byte_count += sum(map(len, lines))
    if not selected:
        raise ValueError('No complete source group fits the upload limits')
    content = b''.join(encoded)
    preview = preview_dataset(template=template, content=content, format='jsonl')
    repository, author, provenance, import_provenance = SOURCES[dataset]
    source_url = 'https://huggingface.co/datasets/' + repository
    manifest = {
        'schema_version': 'rateloop.public-dataset-preparation.v1',
        'source': {'repository': repository, 'revision': revision, 'split': source_split,
                   'export_sha256': source_sha256, 'url': source_url + '/tree/' + revision,
                   'origin_authenticated': False},
        'license': {'id': 'CC-BY-4.0', 'url': 'https://creativecommons.org/licenses/by/4.0/',
                    'attribution': author + ', ' + repository, 'source_url': source_url},
        'conversion': conversion, 'language': language, 'label_provenance': provenance,
        'import_provenance': import_provenance, 'independent_reference': False,
        'template_commitment': commitment(template.model_dump(), 'rateloop.evaluator.template.v1'),
        'output_sha256': hashlib.sha256(content).hexdigest(), 'source_rows': len(raw_rows),
        'row_count': preview['row_count'], 'group_count': preview['group_count'],
        'label_counts': preview['label_counts'], 'excluded_counts': dict(Counter(row['reason'] for row in excluded)),
        'split_policy': 'Import together, then use the existing frozen source-group snapshot ledger; no preassigned split is discarded.',
        'limits': ['Imported public annotations are not independent blind RateLoop references.',
                   'Review source rights and annotations; file hashes identify exports but do not authenticate provenance.',
                   'Exact tokenizer limits are checked before inference/training; no input is truncated.',
                   'Grouping catches source families and formatting duplicates, not every semantic paraphrase.',
                   'Preparation grants no processing, training, sharing or activation permission.'],
    }
    return {'examples': content, 'template': template.model_dump(), 'manifest': manifest,
            'excluded': sorted(excluded, key=lambda row: (row['reason'], row['source_row_sha256']))}


def write_prepared(directory, prepared):
    """Write a new private directory, refusing to overwrite previous evidence."""
    root = Path(directory).expanduser()
    try:
        root.mkdir(mode=0o700, parents=True, exist_ok=False)
    except FileExistsError:
        raise ValueError('Prepared output requires a new directory; previous evidence is never overwritten') from None
    files = {'examples.jsonl': prepared['examples'],
             'template.json': (_json(prepared['template'])+'\n').encode(),
             'manifest.json': (_json(prepared['manifest'])+'\n').encode(),
             'excluded.jsonl': ''.join(_json(row)+'\n' for row in prepared['excluded']).encode()}
    for name, content in files.items():
        fd = os.open(root/name, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, 'O_NOFOLLOW', 0), 0o600)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(content)
    return {'directory': str(root.resolve()), 'rowCount': prepared['manifest']['row_count'],
            'groupCount': prepared['manifest']['group_count'],
            'importProvenance': prepared['manifest']['import_provenance'],
            'independentReference': False, 'trainingStarted': False}
