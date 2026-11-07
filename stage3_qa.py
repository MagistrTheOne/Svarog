#!/usr/bin/env python3
"""Read-only QA: Stage 13 best.pt vs original HF base, held-out CE and generations."""
import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

from stage3_train import TokenCorpus, batch

PROMPTS = [
    ("RU factual", "Москва — столица России. Город расположен"),
    ("RU science", "Искусственный интеллект — это"),
    ("RU technical", "Нейронная сеть обучается с помощью"),
    ("RU narrative", "Однажды утром исследователь обнаружил, что"),
    ("EN factual", "Artificial intelligence is a field of"),
    ("EN science", "A neural network learns by adjusting"),
]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def evaluate(model, corpus, seq, batches, pad, seed):
    rng = random.Random(seed)
    losses = []
    model.eval()
    with torch.inference_mode():
        for _ in range(batches):
            x, y = batch(corpus, seq, 1, rng, pad, "cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_ids=x, labels=y)["loss"]
            value = float(loss)
            if not math.isfinite(value):
                raise RuntimeError("Non-finite validation loss")
            losses.append(value)
    return sum(losses) / len(losses)


def generate(model, tokenizer, prompt, max_new, temperature):
    tokens = tokenizer.encode(prompt)
    if len(tokens) + max_new > model.config.max_position_embeddings:
        raise ValueError("Prompt exceeds context")
    x = torch.tensor([tokens], device="cuda", dtype=torch.long)
    eos = tokenizer.eos_id()
    torch.manual_seed(20261107)
    torch.cuda.manual_seed_all(20261107)
    model.eval()
    result = []
    with torch.inference_mode():
        for _ in range(max_new):
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(input_ids=x)["logits"][0, -1].float()
            logits = logits / temperature
            values, indices = torch.topk(logits, k=40)
            probabilities = torch.softmax(values, dim=-1)
            chosen = indices[torch.multinomial(probabilities, 1)]
            token = int(chosen)
            if token == eos:
                break
            result.append(token)
            x = torch.cat((x, chosen.reshape(1, 1)), dim=1)
    return tokenizer.decode(result)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, default=Path("hf_model"))
    p.add_argument("--checkpoint-dir", type=Path, default=Path("stage3/checkpoints"))
    p.add_argument("--val-batches", type=int, default=128)
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--max-new", type=int, default=72)
    p.add_argument("--output", type=Path, default=Path("stage3/qa_stage13.json"))
    a = p.parse_args()
    if a.val_batches < 1 or a.seq < 16 or a.max_new < 1:
        p.error("Invalid QA parameters")
    if not torch.cuda.is_available():
        p.error("CUDA required")
    best = a.checkpoint_dir / "best.pt"
    latest = a.checkpoint_dir / "latest.pt"
    if not best.is_file() or not latest.is_file():
        p.error("Both best.pt and latest.pt must exist")
    if a.output.exists():
        p.error(f"Refusing to overwrite QA report: {a.output}")
    base_weights = a.model_dir / "model.safetensors"
    tokenizer_path = a.model_dir / "tokenizer.model"
    for path in (base_weights, tokenizer_path):
        if not path.is_file():
            p.error(f"Missing {path}")
    sys.path.insert(0, str(a.model_dir.resolve()))
    from inference import load_svarog

    before = {str(path): digest(path) for path in (best, latest, base_weights)}
    state = torch.load(best, map_location="cpu", weights_only=False)
    if int(state["step"]) != 6000:
        print("WARNING: best checkpoint step:", state["step"])
    val_meta = json.loads((a.checkpoint_dir / "cache_validation" / "meta.json").read_text(encoding="utf-8"))
    if state["fingerprint"]["validation_sha256"] != val_meta["text_sha256"]:
        raise RuntimeError("Validation dataset fingerprint mismatch")
    if state["fingerprint"]["weights_sha256"] != before[str(base_weights)]:
        raise RuntimeError("Base weights fingerprint mismatch")
    if state["fingerprint"]["tokenizer_sha256"] != digest(tokenizer_path):
        raise RuntimeError("Tokenizer fingerprint mismatch")
    if a.seq != state["fingerprint"]["seq"]:
        raise RuntimeError("QA sequence length must match training sequence")
    corpus = TokenCorpus(a.checkpoint_dir / "cache_validation")
    model, tokenizer = load_svarog(a.model_dir.resolve(), device="cpu")
    model = model.to("cuda").eval()
    result = {
        "status": "STARTED",
        "best_step": int(state["step"]),
        "best_val_recorded": float(state["best_val"]),
        "checkpoint_sha256": before,
        "validation_batches": a.val_batches,
        "validation_seed": 2042,
        "seq": a.seq,
        "max_new": a.max_new,
        "generation": [],
    }
    print("NULLXES SVAROG | STAGE 13 | READ-ONLY QA", flush=True)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("Evaluating HF BASE...", flush=True)
    result["base_loss"] = evaluate(model, corpus, a.seq, a.val_batches, tokenizer.pad_id(), 2042)
    result["base_ppl"] = math.exp(result["base_loss"])
    base_generations = []
    for title, prompt in PROMPTS:
        base_generations.append(generate(model, tokenizer, prompt, a.max_new, 0.8))
    print(f"BASE loss={result['base_loss']:.5f} ppl={result['base_ppl']:.2f}", flush=True)

    model.load_state_dict(state["model"], strict=True)
    del state
    model.eval()
    print("Evaluating STAGE 13 BEST...", flush=True)
    result["stage13_loss"] = evaluate(model, corpus, a.seq, a.val_batches, tokenizer.pad_id(), 2042)
    result["stage13_ppl"] = math.exp(result["stage13_loss"])
    result["loss_delta"] = result["stage13_loss"] - result["base_loss"]
    print(f"STAGE13 loss={result['stage13_loss']:.5f} ppl={result['stage13_ppl']:.2f}", flush=True)
    for (title, prompt), baseline in zip(PROMPTS, base_generations):
        trained = generate(model, tokenizer, prompt, a.max_new, 0.8)
        result["generation"].append({"test": title, "prompt": prompt,
                                      "base": baseline, "stage13": trained})
        print("\n" + "=" * 68)
        print(title, "|", prompt)
        print("HF BASE:", baseline)
        print("STAGE 13:", trained)
    after = {str(path): digest(path) for path in (best, latest, base_weights)}
    if before != after:
        raise RuntimeError("Model/checkpoint file unexpectedly changed during QA")
    result["status"] = "COMPLETE"
    result["checkpoint_unchanged"] = True
    a.output.parent.mkdir(parents=True, exist_ok=True)
    temp = a.output.with_suffix(".json.partial")
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(a.output)
    print("\n" + "=" * 68)
    print("QA COMPLETE:", a.output)
    print(f"BASE {result['base_loss']:.5f} -> STAGE13 {result['stage13_loss']:.5f}")
    print("CHECKPOINTS AND HF WEIGHTS: UNCHANGED")


if __name__ == "__main__":
    main()
