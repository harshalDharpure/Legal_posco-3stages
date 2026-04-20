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
    out: list[dict[str, Any]] = []

    # Rolling-window context:
    # Turn1 -> Turn2
    # Turn1+2 -> Turn3
    # ... where "input" is the concatenated dialogue history up to the user turn
    # preceding the target assistant response.
    history: list[str] = []
    turn_i = 0
    for t in turns:
        role = str(t.get("role", "")).strip().lower()
        text = str(t.get("text", "")).strip()
        if not role or not text:
            continue

        if role == "user":
            history.append(f"[USER]: {text}")
        elif role == "assistant":
            # Only create an example if there is some preceding user context.
            if history:
                inp = "\n".join(history) + "\n[ASSISTANT]:"
                ex = {
                    # Keep original metadata if present
                    "dialogue_id": dialogue_row.get("dialogue_id", ""),
                    "language": dialogue_row.get("language", ""),
                    "statutes_cited": dialogue_row.get("statutes_cited", []),
                    "metadata": dialogue_row.get("metadata", {}),
                    # Training pair
                    "input": inp,
                    "output": text,
                    "turn_index": turn_i,
                }
                out.append(ex)
            history.append(f"[ASSISTANT]: {text}")
        else:
            continue
        turn_i += 1
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
        # `input` is already a fully formatted prompt ending with "[ASSISTANT]:"
        # OR a raw user string (pair-level legacy); we normalize to the strict template.
        inp = str(row.get("input", "")).strip()
        assistant = str(row.get("output", "")).strip()
        if "[ASSISTANT]:" in inp and inp.startswith("[USER]:"):
            text = inp + f" {assistant}"
            # For masking, we want labels to start right after the final "[ASSISTANT]: "
            # in this prompt.
            prefix = inp + " "
            plen = len(self.tokenizer(prefix, add_special_tokens=False)["input_ids"])
        else:
            # Legacy pair-level support: treat `input` as user-only.
            text = format_dialogue_prompt(inp, assistant)
            plen = prompt_prefix_tokens_len(self.tokenizer, inp)
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        input_ids = enc["input_ids"]
        attn = enc["attention_mask"]
        plen = min(int(plen), len(input_ids))
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

