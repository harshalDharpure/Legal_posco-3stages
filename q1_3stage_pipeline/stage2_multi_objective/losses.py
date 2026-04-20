"""
Multi-objective losses for Stage 2.

L_gen: causal LM cross-entropy (teacher-forced), prompt masked in labels.

L_entail (default, differentiable): frozen sentence encoder; maximize similarity between
reference answer y+ and pooled hidden states of the assistant span (semantic alignment).
This avoids non-differentiable argmax→DeBERTa while staying faithful to "entailment-style" supervision.

Optional L_entail_deberta: KL to one-hot ENTAILMENT for (premise=y_ref, hypothesis=greedy decode).
Computed with torch.no_grad on NLI weights; greedy decode breaks LM gradients — use only for
logging / auxiliary distillation hooks, not added to backward unless you add a STE/REINFORCE path.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def causal_lm_loss_shifted(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Standard shift: predict token t from position t-1. labels already masked (-100 prompt)."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    return loss


def pooled_assistant_hidden(hidden: torch.Tensor, assistant_mask: torch.Tensor) -> torch.Tensor:
    """Mean pool over assistant tokens. assistant_mask: 1 for assistant positions."""
    mask = assistant_mask.unsqueeze(-1).float()
    summed = (hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


class FrozenSentenceEncoder:
    """Frozen sentence-transformers encoder for entailment/triplet terms."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.encoder = SentenceTransformer(model_name)
        for p in self.encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_texts(self, texts: list[str], device: torch.device) -> torch.Tensor:
        emb = self.encoder.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=False,
            device=str(device),
        )
        return emb.float()


def entailment_cosine_loss(
    pooled_lm: torch.Tensor,
    ref_emb: torch.Tensor,
) -> torch.Tensor:
    """1 - cosine similarity per batch item, mean."""
    p = F.normalize(pooled_lm, dim=-1)
    r = F.normalize(ref_emb, dim=-1)
    cos = (p * r).sum(dim=-1)
    return (1.0 - cos).mean()


def triplet_margin_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = 0.3,
) -> torch.Tensor:
    """Euclidean on normalized vectors: d(a,p) - d(a,n) + margin."""
    a = F.normalize(anchor, dim=-1)
    p = F.normalize(positive, dim=-1)
    n = F.normalize(negative, dim=-1)
    d_pos = (a - p).pow(2).sum(dim=-1)
    d_neg = (a - n).pow(2).sum(dim=-1)
    return F.relu(d_pos - d_neg + margin).mean()


def deberta_entailment_kl(
    nli_model,
    nli_tokenizer,
    premises: list[str],
    hypotheses: list[str],
    device: torch.device,
    entailment_index: int,
) -> torch.Tensor:
    """KL(pred || one_hot(entailment)); NLI forward should be no_grad on nli_model params."""
    enc = nli_tokenizer(
        premises,
        hypotheses,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = nli_model(**enc)
        logits = out.logits
    target = torch.zeros_like(logits)
    target[:, entailment_index] = 1.0
    log_p = F.log_softmax(logits, dim=-1)
    return F.kl_div(log_p, target, reduction="batchmean")

