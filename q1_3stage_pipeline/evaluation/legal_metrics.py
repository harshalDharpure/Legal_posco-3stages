from __future__ import annotations

import re
from typing import Any


_STATUTE_PATTERNS = [
    re.compile(r"\bIPC\s+(\d{1,4})\b", re.IGNORECASE),
    re.compile(r"\bSection\s+(\d{1,4})\b", re.IGNORECASE),
    re.compile(r"\bS\.\s*(\d{1,4})\b", re.IGNORECASE),
]


def extract_statute_ids(text: str) -> set[str]:
    ids: set[str] = set()
    for pat in _STATUTE_PATTERNS:
        for m in pat.finditer(text or ""):
            ids.add(m.group(1))
    return ids


def normalize_statutes_cited(statutes_cited: list[Any] | None) -> set[str]:
    if not statutes_cited:
        return set()
    out: set[str] = set()
    for s in statutes_cited:
        if s is None:
            continue
        out |= extract_statute_ids(str(s))
    return out


def statute_correctness_score(*, statutes_cited: list[Any] | None, candidate: str) -> dict[str, float]:
    """
    Q1-friendly automatic proxy for legal correctness:
    compare statute IDs in generated text vs dataset `statutes_cited`.
    """
    gold = normalize_statutes_cited(statutes_cited)
    pred = extract_statute_ids(candidate)
    if not gold and not pred:
        return {"statute_precision": 1.0, "statute_recall": 1.0, "statute_f1": 1.0}
    if not pred:
        return {"statute_precision": 0.0, "statute_recall": 0.0, "statute_f1": 0.0}
    tp = len(gold & pred)
    prec = tp / max(len(pred), 1)
    rec = tp / max(len(gold), 1)
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {"statute_precision": float(prec), "statute_recall": float(rec), "statute_f1": float(f1)}

