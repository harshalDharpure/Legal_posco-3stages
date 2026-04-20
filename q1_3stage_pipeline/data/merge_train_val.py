#!/usr/bin/env python3
"""Merge train.jsonl + val.jsonl → final_train.jsonl (ONLY after tuning is frozen)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from q1_3stage_pipeline.utils import load_jsonl, save_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="q1_3stage_pipeline/data/splits/train.jsonl")
    ap.add_argument("--val", default="q1_3stage_pipeline/data/splits/val.jsonl")
    ap.add_argument("--out", default="q1_3stage_pipeline/data/splits/final_train.jsonl")
    args = ap.parse_args()

    merged = load_jsonl(args.train) + load_jsonl(args.val)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    save_jsonl(merged, args.out)
    meta = {"train": len(load_jsonl(args.train)), "val": len(load_jsonl(args.val)), "final_train": len(merged)}
    with open(args.out.replace(".jsonl", "_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

