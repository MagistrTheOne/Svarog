"""SVAROG V2: safe, read-only Hugging Face architecture/tokenizer inspection.

No model weights are fetched, no remote Python executed, no token printed.
Usage:
    python hf_bridge_v2.py inspect
    python hf_bridge_v2.py inspect --auth
    python hf_bridge_v2.py compare --local-tokenizer tokenizer/svarog_v2.model

A session token can be supplied with --auth via a hidden getpass prompt,
or through HF_TOKEN in the process environment. Never put it in source code.
"""
import argparse
import getpass
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "MagistrTheOne/NULLXES-KLAREN-4E"
CONFIG_FILES = (
    "config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
)
SAMPLES = (
    "Привет, Сварог! Как дела?",
    "NULLXES разрабатывает автономные интеллектуальные системы.",
    "The quick brown fox jumps over the lazy dog.",
    "def solve(x: int) -> int: return x * 2",
    "Ёжик, 2026. 🚀 Съешь ещё этих мягких французских булок.",
)


def auth_token(interactive: bool) -> str | None:
    existing = os.environ.get("HF_TOKEN")
    if existing:
        return existing
    if interactive:
        return getpass.getpass("Hugging Face session token (hidden): ").strip() or None
    return None


def inspect_repository(api: HfApi):
    info = api.model_info(REPO, files_metadata=False)
    names = sorted(entry.rfilename for entry in (info.siblings or []))
    print(f"Repo: {REPO}")
    print(f"Revision: {info.sha}")
    print(f"Architecture tags: {[tag for tag in info.tags if 'multimodal' in tag or 'gpt' in tag or 'safetensors' in tag]}")
    print(f"Files ({len(names)}):")
    for name in names:
        print(" ", name)
    for name in CONFIG_FILES:
        if name not in names:
            continue
        print(f"\n[{name}]")
        path = hf_hub_download(repo_id=REPO, filename=name, revision=info.sha,
                               token=api.token)
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        if name == "config.json":
            # Config may contain repo-specific paths or irrelevant metadata.
            keys = ("model_type", "architectures", "vocab_size", "hidden_size",
                    "num_hidden_layers", "num_attention_heads", "intermediate_size",
                    "max_position_embeddings", "text_config", "language_config",
                    "vision_config", "audio_config", "tie_word_embeddings",
                    "bos_token_id", "eos_token_id", "pad_token_id")
            document = {k: document[k] for k in keys if k in document}
        print(json.dumps(document, indent=2, ensure_ascii=False)[:16000])
    print("\nTokenizer artifacts:", [n for n in names if
          "tokenizer" in n.lower() or "vocab" in n.lower()
          or "merges" in n.lower() or "special_tokens" in n.lower()])
    return set(names)


def compare(api: HfApi, local_path: Path):
    names = inspect_repository(api)
    if not local_path.exists():
        raise SystemExit(f"Missing local tokenizer: {local_path}. Train it first with tokenizer_v2.py")
    import sentencepiece as spm
    local = spm.SentencePieceProcessor(model_file=str(local_path))
    remote = None
    tokenizer_files = ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")
    if any(x in names for x in tokenizer_files):
        try:
            # Tokenizer only: transformers does not download model weights here.
            from transformers import AutoTokenizer
            remote = AutoTokenizer.from_pretrained(REPO, trust_remote_code=False,
                                                    token=api.token)
        except Exception as exc:
            print("KLAREN tokenizer cannot be loaded safely from this repository:", type(exc).__name__, str(exc)[:250])
            print("Do not enable trust_remote_code merely to bypass this limitation.")
    if remote is None:
        print("KLAREN tokenizer comparison unavailable: tokenization assets may be absent.")
    for text in SAMPLES:
        row = {"text": text, "svarog_tokens": len(local.encode(text)),
               "svarog_roundtrip": local.decode(local.encode(text)) == text}
        if remote is not None:
            ids = remote.encode(text, add_special_tokens=False)
            row["klaren_tokens"] = len(ids)
            row["klaren_roundtrip"] = remote.decode(ids, skip_special_tokens=False) == text
        print(json.dumps(row, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description="Read-only Hugging Face KLAREN 4E inspection")
    p.add_argument("action", choices=("inspect", "compare"))
    p.add_argument("--auth", action="store_true", help="Prompt for session token without echo")
    p.add_argument("--local-tokenizer", type=Path, default=Path("tokenizer/svarog_v2.model"))
    a = p.parse_args()
    token = auth_token(a.auth)
    api = HfApi(token=token)
    if a.action == "inspect":
        inspect_repository(api)
    else:
        compare(api, a.local_tokenizer)


if __name__ == "__main__":
    main()
