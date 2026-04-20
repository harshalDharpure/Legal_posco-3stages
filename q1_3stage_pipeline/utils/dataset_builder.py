from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class SFTExample:
    prompt: str  # ends with "\n[ASSISTANT]:"
    target: str  # assistant response (no prefix)
    dialogue_id: str
    turn_index: int
    language: str
    statutes_cited: list[Any]
    metadata: dict[str, Any]


def _iter_dialogue_sft_examples(row: dict[str, Any]) -> Iterable[SFTExample]:
    """
    Rolling-window construction over `turns`:
      Turn1 -> Turn2
      Turn1+2 -> Turn3
    where each assistant turn becomes a supervised (prompt -> next assistant).

    Prompt format (STRICT, global):
      [USER]: ...
      [ASSISTANT]: ...
      ...
      [USER]: ...
      [ASSISTANT]:
    """
    dialogue_id = str(row.get("dialogue_id", ""))
    language = str(row.get("language", ""))
    statutes = list(row.get("statutes_cited", []) or [])
    metadata = dict(row.get("metadata", {}) or {})

    turns = row.get("turns", []) or []
    history: list[str] = []
    turn_i = 0

    for t in turns:
        role = str(t.get("role", "")).strip().lower()
        text = str(t.get("text", "")).strip()
        if not role or not text:
            turn_i += 1
            continue

        if role == "user":
            history.append(f"[USER]: {text}")
        elif role == "assistant":
            if history:
                prompt = "\n".join(history) + "\n[ASSISTANT]:"
                yield SFTExample(
                    prompt=prompt,
                    target=text,
                    dialogue_id=dialogue_id,
                    turn_index=turn_i,
                    language=language,
                    statutes_cited=statutes,
                    metadata=metadata,
                )
            history.append(f"[ASSISTANT]: {text}")

        turn_i += 1


class DatasetBuilder:
    """
    Master dataset schema (ONLY source of truth):
      {
        "dialogue_id": "...",
        "language": "...",
        "turns": [{"role": "user"|"assistant", "text": "..."}...],
        "statutes_cited": [...],
        "metadata": {...}
      }

    This builder produces derived training examples *in-memory* for each stage.
    It does not materialize separate datasets (beyond strict split JSONLs).
    """

    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def build_sft(self) -> list[dict[str, Any]]:
        """
        Returns a flat list of pair-level examples with:
          - prompt: strict prompt ending with "\n[ASSISTANT]:"
          - output: ground truth assistant response
        """
        out: list[dict[str, Any]] = []
        for r in self.rows:
            if isinstance(r, dict) and "turns" in r:
                for ex in _iter_dialogue_sft_examples(r):
                    out.append(
                        {
                            "dialogue_id": ex.dialogue_id,
                            "language": ex.language,
                            "turn_index": ex.turn_index,
                            "statutes_cited": ex.statutes_cited,
                            "metadata": ex.metadata,
                            "prompt": ex.prompt,
                            "output": ex.target,
                        }
                    )
            else:
                # Pair-level legacy support
                prompt = str(r.get("prompt", "")).strip()
                if not prompt:
                    inp = str(r.get("input", "")).strip()
                    prompt = f"[USER]: {inp}\n[ASSISTANT]:"
                out.append(
                    {
                        "dialogue_id": str(r.get("dialogue_id", "")),
                        "language": str(r.get("language", "")),
                        "turn_index": int(r.get("turn_index", 0) or 0),
                        "statutes_cited": list(r.get("statutes_cited", []) or []),
                        "metadata": dict(r.get("metadata", {}) or {}),
                        "prompt": prompt,
                        "output": str(r.get("output", "")).strip(),
                    }
                )
        return out

    def build_triplets(self) -> list[dict[str, Any]]:
        """
        Returns (x, y_pos) only.
        Negatives MUST be generated dynamically at training time (not stored).
        """
        sft = self.build_sft()
        return [
            {
                "dialogue_id": ex["dialogue_id"],
                "language": ex["language"],
                "turn_index": ex["turn_index"],
                "statutes_cited": ex["statutes_cited"],
                "metadata": ex["metadata"],
                "x": ex["prompt"],
                "y_pos": ex["output"],
            }
            for ex in sft
        ]

    def build_dpo(self) -> list[dict[str, Any]]:
        """
        Returns (prompt, chosen) only.
        Rejected answers MUST be generated dynamically at training time (not stored).
        """
        sft = self.build_sft()
        return [
            {
                "dialogue_id": ex["dialogue_id"],
                "language": ex["language"],
                "turn_index": ex["turn_index"],
                "statutes_cited": ex["statutes_cited"],
                "metadata": ex["metadata"],
                "prompt": ex["prompt"],
                "chosen": ex["output"],
            }
            for ex in sft
        ]

