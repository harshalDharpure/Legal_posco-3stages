#!/usr/bin/env python3
"""Run Stage 2 ablations sequentially (gen_only, gen_entail, gen_triplet, full)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="q1_3stage_pipeline/configs/pipeline_default.yaml")
    ap.add_argument("--train-jsonl", default="q1_3stage_pipeline/data/splits/train.jsonl")
    ap.add_argument("--init-from", choices=["base", "exp3"], default="base")
    ap.add_argument("--out-root", default="q1_3stage_pipeline/logs/checkpoints/stage2_ablations")
    args = ap.parse_args()

    modes = ["gen_only", "gen_entail", "gen_triplet", "full"]
    for m in modes:
        out = _REPO / args.out_root / f"{args.init_from}_{m}"
        out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(_REPO / "q1_3stage_pipeline/stage2_multi_objective/train.py"),
            "--config",
            str(_REPO / args.config),
            "--init-from",
            args.init_from,
            "--ablation",
            m,
            "--train-jsonl",
            str(_REPO / args.train_jsonl),
            "--output-dir",
            str(out),
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=str(_REPO), env={**dict(os.environ), "PYTHONPATH": str(_REPO)})


if __name__ == "__main__":
    main()

