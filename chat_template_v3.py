"""SVAROG V3 chat serialization on the EXISTING SentencePiece vocabulary.

Chat delimiters intentionally use plain text; they are NOT new special-token IDs.
This keeps all trained embeddings and the step-3000 checkpoint compatible.
The format requires supervised fine-tuning before it is useful for answering.
"""
from __future__ import annotations

START = "[im_start]"
END = "[im_end]"
ALLOWED_ROLES = {"system", "user", "assistant"}
DEFAULT_SYSTEM = (
    "Ты Сварог, языковая модель NULLXES. "
    "Отвечай на языке пользователя точно, кратко и по существу. "
    "Если ответа не знаешь, скажи об этом. Не придумывай факты."
)


def format_chat(messages: list[dict], add_generation_prompt: bool = False) -> str:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    pieces = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role not in ALLOWED_ROLES:
            raise ValueError(f"Unsupported role: {role!r}")
        if not isinstance(content, str):
            raise ValueError("Message content must be text")
        if START in content or END in content:
            raise ValueError("Message contains reserved chat delimiters")
        pieces.append(f"{START}{role}\n{content.strip()}{END}\n")
    if add_generation_prompt:
        if messages[-1]["role"] == "assistant":
            raise ValueError("Last role cannot be assistant with generation prompt")
        pieces.append(f"{START}assistant\n")
    return "".join(pieces)


def prompt_for_user(user_text: str, system: str = DEFAULT_SYSTEM) -> str:
    return format_chat(
        [{"role": "system", "content": system},
         {"role": "user", "content": user_text}],
        add_generation_prompt=True,
    )
