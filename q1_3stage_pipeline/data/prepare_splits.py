#!/usr/bin/env python3
"""
Create train / val / test JSONL splits with a fixed seed (no overlap, reproducible).

Usage:
  python q1_3stage_pipeline/data/prepare_splits.py \
    --source q1_3stage_pipeline/data/raw/train.jsonl \
    --out-dir q1_3stage_pipeline/data/splits \
    --ratios 0.8 0.1 0.1 \
    --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from q1_3stage_pipeline.utils import save_jsonl


def load_all_lines(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="Input JSONL (must have input/output)")
    p.add_argument("--out-dir", default="q1_3stage_pipeline/data/splits")
    p.add_argument("--ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1], metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    r_train, r_val, r_test = args.ratios
    if abs(r_train + r_val + r_test - 1.0) > 1e-6:
        raise SystemExit("ratios must sum to 1.0")

    random.seed(args.seed)
    rows = load_all_lines(args.source)
    random.shuffle(rows)

    n = len(rows)
    n_train = int(n * r_train)
    n_val = int(n * r_val)

    train = rows[:n_train]
    val = rows[n_train : n_train + n_val]
    test = rows[n_train + n_val :]

    os.makedirs(args.out_dir, exist_ok=True)
    meta = {
        "source": args.source,
        "seed": args.seed,
        "ratios": [r_train, r_val, r_test],
        "counts": {"train": len(train), "val": len(val), "test": len(test), "total": n},
    }
    save_jsonl(train, os.path.join(args.out_dir, "train.jsonl"))
    save_jsonl(val, os.path.join(args.out_dir, "val.jsonl"))
    save_jsonl(test, os.path.join(args.out_dir, "test.jsonl"))
    with open(os.path.join(args.out_dir, "split_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

