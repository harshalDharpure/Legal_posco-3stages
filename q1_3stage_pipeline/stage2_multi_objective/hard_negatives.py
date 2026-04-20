"""Resolve y- (hard negative) for triplet loss."""

from __future__ import annotations

import random
from typing import Any


def get_negative_output(
    batch_rows: list[dict[str, Any]],
    index: int,
    rng: random.Random,
) -> str:
    """
    Priority:
    1. Field `negative_output` in the row (manual / mined).
    2. In-batch random other `output` (wrong answer proxy).
    """
    row = batch_rows[index]
    neg = row.get("negative_output") or row.get("hard_negative")
    if neg and str(neg).strip():
        return str(neg).strip()
    if len(batch_rows) <= 1:
        return "(No alternative answer available.)"
    j = rng.randint(0, len(batch_rows) - 1)
    if j == index:
        j = (j + 1) % len(batch_rows)
    return str(batch_rows[j].get("output", "")).strip()

