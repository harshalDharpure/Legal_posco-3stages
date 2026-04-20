"""SFT examples with prompt tokens masked in labels (-100)."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from q1_3stage_pipeline.utils import format_dialogue_prompt, prompt_prefix_tokens_len


def _dialogue_to_examples(dialogue_row: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Convert one dialogue-level row with `turns` into (input, output) examples in-memory.
    This does NOT write any new files; it is only used at training time.
    """
    turns = dialogue_row.get("turns", [])
    user_queries = [t.get("text", "") for t in turns if t.get("role") == "user"]
    assistant_responses = [t.get("text", "") for t in turns if t.get("role") == "assistant"]
    n = min(len(user_queries), len(assistant_responses))
    out: list[dict[str, Any]] = []
    for i in range(n):
        ex = {
            # Keep original metadata if present
            "dialogue_id": dialogue_row.get("dialogue_id", ""),
            "language": dialogue_row.get("language", ""),
            "complexity": dialogue_row.get("complexity", ""),
            "bucket": dialogue_row.get("bucket", ""),
            "case_id": dialogue_row.get("case_id", 0),
            "statutes_cited": dialogue_row.get("statutes_cited", []),
            # Training pair
            "input": str(user_queries[i]).strip(),
            "output": str(assistant_responses[i]).strip(),
            "turn_index": i + 1,
        }
        # Carry negatives if someone pre-attached them at dialogue level (optional).
        if "negative_output" in dialogue_row:
            ex["negative_output"] = dialogue_row.get("negative_output")
        if "hard_negative" in dialogue_row:
            ex["hard_negative"] = dialogue_row.get("hard_negative")
        out.append(ex)
    return out


def _flatten_rows_to_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Accepts either:
    - pair-level rows with fields `input`/`output`, OR
    - dialogue-level rows with field `turns` (list of role/text dicts).
    Returns a flat list of pair-level examples.
    """
    examples: list[dict[str, Any]] = []
    for r in rows:
        if isinstance(r, dict) and "turns" in r and ("input" not in r or "output" not in r):
            examples.extend(_dialogue_to_examples(r))
        else:
            examples.append(r)
    return examples


class LegalSFTDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer,
        max_length: int,
        return_row_index: bool = False,
    ):
        # Keep a flattened list of (input, output) examples for training.
        self.examples = _flatten_rows_to_examples(rows)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_row_index = return_row_index

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.examples[idx]
        user = row.get("input", "").strip()
        assistant = row.get("output", "").strip()
        text = format_dialogue_prompt(user, assistant)
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        input_ids = enc["input_ids"]
        attn = enc["attention_mask"]
        plen = min(prompt_prefix_tokens_len(self.tokenizer, user), len(input_ids))
        labels = input_ids.copy()
        for i in range(plen):
            labels[i] = -100
        out = {
            "input_ids": input_ids,
            "attention_mask": attn,
            "labels": labels,
        }
        if self.return_row_index:
            # Index into the flattened examples list (stable for this dataset instance).
            out["_row_index"] = idx
        return out


def collate_sft_batch(batch: list[dict[str, Any]], tokenizer) -> dict[str, torch.Tensor | list[int]]:
    """Pad sequences; mask pad tokens in labels."""
    max_len = max(len(x["input_ids"]) for x in batch)
    pad_id = tokenizer.pad_token_id
    input_ids = []
    attn = []
    labels = []
    indices: list[int] = []
    for x in batch:
        ids = x["input_ids"]
        m = x["attention_mask"]
        lab = x["labels"]
        extra = max_len - len(ids)
        input_ids.append(ids + [pad_id] * extra)
        attn.append(m + [0] * extra)
        labels.append(lab + [-100] * extra)
        if "_row_index" in x:
            indices.append(int(x["_row_index"]))
    batch_tensors: dict[str, torch.Tensor | list[int]] = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
    if indices:
        batch_tensors["_row_indices"] = indices
    return batch_tensors

