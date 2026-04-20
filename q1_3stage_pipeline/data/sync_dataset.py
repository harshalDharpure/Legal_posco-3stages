#!/usr/bin/env python3
"""
Copy (sync) the dataset files used by the 3-stage pipeline into this standalone folder.

Example:
  python q1_3stage_pipeline/data/sync_dataset.py \
    --source-dir experiments/exp3_pretraining_finetuning/finetuning \
    --out-dir q1_3stage_pipeline/data/raw
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", required=True, help="Folder containing train.jsonl/val.jsonl/test.jsonl")
    ap.add_argument("--out-dir", required=True, help="Target folder inside q1_3stage_pipeline/")
    args = ap.parse_args()

    src_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in ["train.jsonl", "val.jsonl", "test.jsonl"]:
        src = src_dir / name
        if not src.is_file():
            raise SystemExit(f"Missing {name} in {src_dir}")
        dst = out_dir / name
        shutil.copyfile(src, dst)
        print(f"Copied {src} -> {dst} ({os.path.getsize(dst)} bytes)")


if __name__ == "__main__":
    main()

