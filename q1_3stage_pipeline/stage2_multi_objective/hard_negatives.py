"""Dynamic hard-negative generation + filtering for triplet/DPO."""

from __future__ import annotations

import random
from typing import Any

import torch


_IPC_CORRUPTIONS = {
    "IPC 376": "IPC 420",
    "Section 376": "Section 420",
    "S. 376": "S. 420",
    "376 IPC": "420 IPC",
    "non-compoundable": "compoundable",
    "compoundable": "non-compoundable",
    "minor": "major",
    "major": "minor",
}


def corrupt_legal_text(text: str) -> str:
    out = text
    for a, b in _IPC_CORRUPTIONS.items():
        out = out.replace(a, b)
    return out


def cross_sample_negative(all_rows: list[dict[str, Any]], rng: random.Random, avoid_dialogue_id: str) -> str:
    """Take assistant response from another dialogue (cross-sample negative)."""
    if not all_rows:
        return ""
    for _ in range(10):
        r = all_rows[rng.randrange(len(all_rows))]
        if str(r.get("dialogue_id", "")) != avoid_dialogue_id:
            return str(r.get("output", "")).strip()
    return str(all_rows[rng.randrange(len(all_rows))].get("output", "")).strip()


@torch.no_grad()
def model_negative_generate(
    model,
    tokenizer,
    prompt: str,
    device,
    rng: random.Random,
    *,
    max_new_tokens: int = 128,
) -> str:
    """
    Model negative (on-the-fly):
      y_neg1 = model.generate(x, temperature=0.7, top_p=0.9)
    """
    import torch

    # Make generation reproducible per step when seed is set globally.
    torch.manual_seed(rng.randrange(2**31 - 1))

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_ids = out[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def select_hard_negative(
    *,
    x: str,
    y_pos: str,
    candidates: list[str],
    sentence_encoder,
    sim_low_margin: float = 0.0,
    sim_high_threshold: float = 0.2,
    # Semi-hard mining: keep negatives close to the positive, but still distinct.
    # If None, no upper bound is applied (backwards compatible).
    sim_pos_gap_min: float = 0.0,
    sim_pos_gap_max: float | None = None,
) -> str:
    """
    Hard-negative filtering (frozen Sentence-BERT):
      keep negative only if sim(x, y-) < sim(x, y+) AND sim(x,y-) still high
    Then hard mining:
      y_neg = argmax sim(x, candidate_negatives)
    """
    import torch
    import torch.nn.functional as F

    cands = [c.strip() for c in candidates if c and c.strip()]
    if not cands:
        return "(No negative available.)"

    with torch.no_grad():
        # Force SBERT similarity computation on CPU to avoid VRAM contention.
        x_emb = sentence_encoder.encode([x], convert_to_tensor=True, show_progress_bar=False, device="cpu")
        pos_emb = sentence_encoder.encode([y_pos], convert_to_tensor=True, show_progress_bar=False, device="cpu")
        cand_emb = sentence_encoder.encode(cands, convert_to_tensor=True, show_progress_bar=False, device="cpu")

        x_emb = F.normalize(x_emb.float(), dim=-1)
        pos_emb = F.normalize(pos_emb.float(), dim=-1)
        cand_emb = F.normalize(cand_emb.float(), dim=-1)

        sim_pos = float((x_emb * pos_emb).sum(dim=-1).item())
        sims = (cand_emb @ x_emb.squeeze(0)).squeeze(-1)

    kept: list[tuple[float, str]] = []
    for sim, txt in zip(sims.tolist(), cands):
        # Must be hard-ish: sufficiently similar to x, but less similar than the positive.
        # Optionally restrict to a "semi-hard" band relative to sim_pos.
        if sim < (sim_pos - sim_low_margin) and sim >= sim_high_threshold:
            gap = sim_pos - float(sim)
            if gap >= float(sim_pos_gap_min) and (sim_pos_gap_max is None or gap <= float(sim_pos_gap_max)):
                kept.append((float(sim), txt))
    if not kept:
        # fallback: best non-pos candidate (still encourages "hard")
        j = int(sims.argmax().item())
        return cands[j]
    kept.sort(key=lambda t: t[0], reverse=True)
    return kept[0][1]


