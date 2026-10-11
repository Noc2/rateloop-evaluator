"""Stable Unicode14 duplicate detection for every supported Python runtime.

NFKC is stable for this assigned repertoire. Newer characters remain opaque;
this is a duplicate guard, not semantic similarity or a text display transform.
The adjacent table contains official source digests and the Unicode data license.
"""
from bisect import bisect_right
import json
from pathlib import Path
import re
import unicodedata

_DATA = json.loads(Path(__file__).with_name("unicode14.json").read_text())
_RANGES = _DATA["unassignedRanges"]
_STARTS = [pair[0] for pair in _RANGES]
_FOLDS = _DATA["caseFolds"]
_SPACES = re.compile("[" + "".join(chr(cp) for cp in _DATA["whitespaceCodePoints"]) + "]+")


def normalize_learning_text(value: str) -> str:
    chunks = []
    run = []
    for char in value:
        cp = ord(char)
        index = bisect_right(_STARTS, cp) - 1
        if index >= 0 and cp <= _RANGES[index][1]:
            chunks.extend((unicodedata.normalize("NFKC", "".join(run)), char))
            run.clear()
        else:
            run.append(char)
    chunks.append(unicodedata.normalize("NFKC", "".join(run)))
    folded = "".join(_FOLDS.get(char, char) for char in "".join(chunks))
    return _SPACES.sub(" ", folded).strip(" ")
