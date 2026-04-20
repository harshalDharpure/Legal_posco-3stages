"""Single formatting contract across all stages (User/Assistant template)."""

from __future__ import annotations


def format_dialogue_prompt(user_text: str, assistant_text: str | None = None) -> str:
    p = f"User: {user_text}\nAssistant:"
    if assistant_text is not None:
        p += f" {assistant_text}"
    return p


def prompt_prefix_tokens_len(tokenizer, user_text: str) -> int:
    """Length in tokens of prefix up to and including 'Assistant: ' (labels start after this)."""
    prefix = f"User: {user_text}\nAssistant: "
    return len(tokenizer(prefix, add_special_tokens=False)["input_ids"])

