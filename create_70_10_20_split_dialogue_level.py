"""
Create 70/10/20 train/val/test split at the *dialogue level* (NO pair creation).

Input:
  - hindi_complete_posco_data.jsonl
  - english_posco_dataset.jsonl
  - code_mixed_posco_dataset.jsonl

Output (dialogue-level JSONL, same schema as raw):
  data/splits_dialogue_level/train_70_dialogues.jsonl
  data/splits_dialogue_level/val_10_dialogues.jsonl
  data/splits_dialogue_level/test_20_dialogues.jsonl

Stratified by:
  - language
  - complexity
  - bucket
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from typing import Dict, List


def load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def get_strata_key_dialogue(entry: Dict) -> str:
    return f"{entry.get('language','')}_{entry.get('complexity','')}_{entry.get('bucket','')}"


def main() -> None:
    random.seed(42)

    raw_files = [
        "hindi_complete_posco_data.jsonl",
        "english_posco_dataset.jsonl",
        "code_mixed_posco_dataset.jsonl",
    ]
    dialogues: List[Dict] = []
    for p in raw_files:
        if not os.path.isfile(p):
            raise SystemExit(f"Missing raw dataset file: {p}")
        dialogues.extend(load_jsonl(p))

    print(f"Loaded dialogue samples: {len(dialogues)}")

    strata = defaultdict(list)
    for d in dialogues:
        strata[get_strata_key_dialogue(d)].append(d)
    print(f"Unique strata: {len(strata)}")

    # We want *exact* global totals (840/120/240 for 1200 dialogues),
    # while still stratifying across keys.
    total_n = len(dialogues)
    target_train = int(total_n * 0.7)
    target_val = int(total_n * 0.1)
    target_test = total_n - target_train - target_val

    # Shuffle within each stratum once.
    for items in strata.values():
        random.shuffle(items)

    # Base allocation with floors; put all remainder into test.
    alloc: dict[str, list[int]] = {}
    frac_train: list[tuple[float, str]] = []
    frac_val: list[tuple[float, str]] = []

    for key, items in strata.items():
        n = len(items)
        ideal_tr = n * 0.7
        ideal_va = n * 0.1
        a_tr = int(ideal_tr)  # floor
        a_va = int(ideal_va)  # floor
        a_te = n - a_tr - a_va
        alloc[key] = [a_tr, a_va, a_te]
        frac_train.append((ideal_tr - a_tr, key))
        frac_val.append((ideal_va - a_va, key))

    # Distribute remaining train slots by taking from test in strata
    # with largest fractional remainder for train.
    cur_train = sum(v[0] for v in alloc.values())
    need_train = target_train - cur_train
    frac_train.sort(reverse=True)
    i = 0
    while need_train > 0 and i < len(frac_train):
        _, key = frac_train[i]
        if alloc[key][2] > 0:
            alloc[key][0] += 1
            alloc[key][2] -= 1
            need_train -= 1
        i += 1

    # Distribute remaining val slots similarly (from test) by val remainder.
    cur_val = sum(v[1] for v in alloc.values())
    need_val = target_val - cur_val
    frac_val.sort(reverse=True)
    i = 0
    while need_val > 0 and i < len(frac_val):
        _, key = frac_val[i]
        if alloc[key][2] > 0:
            alloc[key][1] += 1
            alloc[key][2] -= 1
            need_val -= 1
        i += 1

    # Sanity: exact totals must match now.
    cur_train = sum(v[0] for v in alloc.values())
    cur_val = sum(v[1] for v in alloc.values())
    cur_test = sum(v[2] for v in alloc.values())
    if (cur_train, cur_val, cur_test) != (target_train, target_val, target_test):
        raise SystemExit(
            f"Allocation mismatch: got train/val/test={cur_train}/{cur_val}/{cur_test} "
            f"expected {target_train}/{target_val}/{target_test}"
        )

    # Materialize splits using the per-stratum allocations.
    train: List[Dict] = []
    val: List[Dict] = []
    test: List[Dict] = []
    for key, items in strata.items():
        a_tr, a_va, a_te = alloc[key]
        train.extend(items[:a_tr])
        val.extend(items[a_tr : a_tr + a_va])
        test.extend(items[a_tr + a_va : a_tr + a_va + a_te])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    out_dir = "data/splits_dialogue_level"
    save_jsonl(train, os.path.join(out_dir, "train_70_dialogues.jsonl"))
    save_jsonl(val, os.path.join(out_dir, "val_10_dialogues.jsonl"))
    save_jsonl(test, os.path.join(out_dir, "test_20_dialogues.jsonl"))

    total = len(train) + len(val) + len(test)
    print("Split sizes:")
    print(f"  train: {len(train)}")
    print(f"  val:   {len(val)}")
    print(f"  test:  {len(test)}")
    print(f"  total: {total}")


if __name__ == "__main__":
    main()

