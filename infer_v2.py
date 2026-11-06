"""SVAROG V2 — inference from a native train_v2.py checkpoint.

Examples:
  python infer_v2.py --prompt "Сварог — это" --max-new-tokens 48
  python infer_v2.py --prompt "Искусственный интеллект" --greedy
  python infer_v2.py --prompt "The future of robotics" --temperature 0.9 --top-p 0.9

Only load checkpoints you created or trust: PyTorch checkpoints can contain
pickle objects. No remote code, model downloads, or dependency installs.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import sentencepiece as spm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from model import SvarogConfig, SvarogForCausalLM


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description="Generate text with trained Svarog V2")
    p.add_argument("--config", type=Path, default=Path("config_v2_2080.json"))
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/svarog_v2_2080/latest.pt"))
    p.add_argument("--prompt", type=str, default="Сварог — это")
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = p.parse_args()
    if not args.prompt:
        p.error("Prompt must not be empty")
    if args.max_new_tokens < 1:
        p.error("max-new-tokens must be positive")
    if args.temperature <= 0 or not 0 < args.top_p <= 1:
        p.error("temperature must be > 0 and top-p must be in (0,1]")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    model_cfg = SvarogConfig(**config["model"])
    tokenizer_path = Path(config["tokenizer"]["model_path"])
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    if tokenizer.vocab_size() != model_cfg.vocab_size:
        p.error("Tokenizer/model vocabulary mismatch")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device)
    if device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA is not available")
    # weights_only=False is needed for optimizer/RNG objects in this checkpoint.
    # Load only a checkpoint you created or explicitly trust.
    try:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except FileNotFoundError:
        p.error(f"Checkpoint not found: {args.checkpoint}")
    fingerprints = state.get("fingerprints", {})
    for key, path in (("config", args.config), ("tokenizer", tokenizer_path)):
        if fingerprints.get(key) != digest(path):
            p.error(f"Checkpoint {key} fingerprint mismatch. Use its original files.")
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model = SvarogForCausalLM(model_cfg)
    model.load_state_dict(state["model"], strict=True)
    model = model.to(device).eval()
    ids = [tokenizer.bos_id()] + tokenizer.encode(args.prompt, out_type=int)
    max_context = model_cfg.max_position_embeddings
    if len(ids) >= max_context:
        p.error(f"Prompt exceeds model context ({max_context} tokens)")
    print(f"SVAROG V2 | step={state.get('step', '?')} | device={device} | "
          f"parameters={sum(p.numel() for p in model.parameters()):,}")
    print(f"Prompt tokens: {len(ids)} | sampling: {'greedy' if args.greedy else 'top-p'}")
    new_ids = []
    with torch.inference_mode():
        for _ in range(args.max_new_tokens):
            context = torch.tensor([ids[-max_context:]], dtype=torch.long, device=device)
            if device == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(context)["logits"][0, -1].float()
            else:
                logits = model(context)["logits"][0, -1].float()
            # Do not emit training/control tokens as ordinary text.
            logits[tokenizer.pad_id()] = -float("inf")
            logits[tokenizer.bos_id()] = -float("inf")
            if args.greedy:
                next_id = int(logits.argmax())
            else:
                logits /= args.temperature
                values, indices = torch.sort(logits, descending=True)
                probabilities = torch.softmax(values, dim=-1)
                drop = probabilities.cumsum(dim=-1) - probabilities > args.top_p
                values.masked_fill_(drop, -float("inf"))
                sampled = torch.multinomial(torch.softmax(values, dim=-1), 1)
                next_id = int(indices[sampled].item())
            if next_id == tokenizer.eos_id():
                break
            ids.append(next_id)
            new_ids.append(next_id)
    continuation = tokenizer.decode(new_ids)
    print("\n--- PROMPT ---\n" + args.prompt)
    print("\n--- GENERATED CONTINUATION ---\n" + continuation)
    print(f"\nGenerated tokens: {len(new_ids)}")
    print("\nNOTE: 100 training steps on a small corpus are not enough for coherent language.")


if __name__ == "__main__":
    main()
