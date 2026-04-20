"""Single formatting contract across all stages (User/Assistant template)."""

from __future__ import annotations


def format_dialogue_prompt(user_text: str, assistant_text: str | None = None) -> str:
    # STRICT global template (must match across all stages):
    # [USER]: {input}
    # [ASSISTANT]:
    p = f"[USER]: {user_text}\n[ASSISTANT]:"
    if assistant_text is not None:
        p += f" {assistant_text}"
    return p


def prompt_prefix_tokens_len(tokenizer, user_text: str) -> int:
    """Length in tokens of prefix up to and including 'Assistant: ' (labels start after this)."""
    prefix = f"[USER]: {user_text}\n[ASSISTANT]: "
    return len(tokenizer(prefix, add_special_tokens=False)["input_ids"])

