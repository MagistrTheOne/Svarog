#!/usr/bin/env python3
"""Stage 18F: GPT-6-authored instruction curriculum from Stage 17 best.pt.

All labels for system/user/chat headers are -100. The model internally shifts
labels by one, so the first assistant token is learned from the preceding prompt.
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch

from stage3_train import sha256, save_state


def encode_record(row, sp, seq):
    messages = row["messages"]
    if [m["role"] for m in messages] != ["system", "user", "assistant"]:
        raise ValueError("Expected system/user/assistant turn")
    for m in messages:
        if "[im_start]" in m["content"] or "[im_end]" in m["content"]:
            raise ValueError("Reserved delimiter in content")
    # Encode full prefix and answer separately to keep mask aligned. SentencePiece
    # may change token boundaries across concatenation; each piece is decoded
    # independently and the same construction is used during inference.
    prefix = "".join(f"[im_start]{m['role']}\n{m['content'].strip()}[im_end]\n"
                     for m in messages[:-1]) + "[im_start]assistant\n"
    answer = messages[-1]["content"].strip() + "[im_end]\n"
    prefix_ids = [sp.bos_id()] + sp.encode(prefix, out_type=int)
    answer_ids = sp.encode(answer, out_type=int) + [sp.eos_id()]
    # Never truncate user/system content: reject overlong examples instead.
    if len(prefix_ids) + len(answer_ids) > seq or len(answer_ids) < 4:
        return None
    ids = prefix_ids + answer_ids
    labels = [-100] * len(prefix_ids) + answer_ids
    assert len(ids) == len(labels)
    assert labels[len(prefix_ids) - 1] == -100
    assert labels[len(prefix_ids)] == ids[len(prefix_ids)]
    return ids, labels


def load_data(path, sp, seq):
    data = []
    dropped = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            sample = encode_record(row, sp, seq)
            if sample is None:
                dropped += 1
            else:
                data.append(sample)
    print(f"DATA {path}: kept={len(data):,} too_long={dropped:,}", flush=True)
    if not data:
        raise RuntimeError("No examples fit the sequence length")
    return data


def make_batch(data, indices, pad, seq, device):
    x = torch.full((len(indices), seq), pad, dtype=torch.long)
    y = torch.full((len(indices), seq), -100, dtype=torch.long)
    for row, i in enumerate(indices):
        ids, labels = data[i]
        x[row, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        y[row, :len(labels)] = torch.tensor(labels, dtype=torch.long)
    return x.to(device), y.to(device)


def evaluate(model, data, pad, seq, n):
    model.eval()
    total_loss, total_tokens = 0., 0
    with torch.inference_mode():
        for start in range(0, min(n, len(data))):
            x, y = make_batch(data, [start], pad, seq, "cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_ids=x, labels=y)["loss"]
            ntokens = int((y[:, 1:] != -100).sum())
            if not math.isfinite(float(loss)):
                raise RuntimeError("Nonfinite validation loss")
            total_loss += float(loss) * ntokens
            total_tokens += ntokens
    model.train()
    return total_loss / total_tokens


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, default=Path("hf_model"))
    p.add_argument("--stage17-best", type=Path, default=Path("stage5/checkpoints/best.pt"))
    p.add_argument("--data-dir", type=Path, default=Path("data_stage18f_addition"))
    p.add_argument("--output", type=Path, default=Path("stage8_addition/checkpoints"))
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-6)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--val-batches", type=int, default=75)
    p.add_argument("--seed", type=int, default=731)
    p.add_argument("--arm", choices=["direct","step"], required=True)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    if min(a.steps, a.seq, a.batch, a.accum, a.eval_every, a.save_every, a.val_batches) < 1 or a.lr <= 0:
        p.error("Invalid hyperparameters")
    if not torch.cuda.is_available():
        p.error("CUDA required")
    import sentencepiece as spm
    tokenizer_file = a.model_dir / "tokenizer.model"
    if not tokenizer_file.exists() or not a.stage17_best.exists():
        p.error("Missing tokenizer or Stage 17 best.pt")
    train_file = a.data_dir / f"{a.arm}_train.jsonl"
    val_file = a.data_dir / f"{a.arm}_validation.jsonl"
    if not train_file.exists() or not val_file.exists():
        p.error("Stage 18F arithmetic corpus missing. Run stage18f_prepare.py first")
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_file))
    if sp.vocab_size() != 24000:
        p.error("Wrong tokenizer")
    train = load_data(train_file, sp, a.seq)
    validation = load_data(val_file, sp, a.seq)
    if len(validation) < 32:
        p.error("Validation dataset too small")
    a.output = a.output / a.arm
    a.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(a.model_dir.resolve()))
    from inference import load_svarog
    model, _ = load_svarog(a.model_dir.resolve(), device="cpu")
    if a.seq > model.config.max_position_embeddings:
        p.error("Sequence exceeds model context")
    stage17 = torch.load(a.stage17_best, map_location="cpu", weights_only=False)
    if stage17.get("step") != 400:
        print("WARNING: Stage 17 best step is", stage17.get("step"))
    model.load_state_dict(stage17["model"], strict=True)
    del stage17
    model = model.to("cuda").train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    fingerprint = {
        "stage17_sha256": sha256(a.stage17_best),
        "tokenizer_sha256": sha256(tokenizer_file),
        "train_sha256": sha256(train_file),
        "validation_sha256": sha256(val_file),
        "arm": a.arm, "seq": a.seq, "batch": a.batch, "accum": a.accum,
        "lr": a.lr, "seed": a.seed
    }
    latest, best_file = a.output / "latest.pt", a.output / "best.pt"
    start, best = 0, float("inf")
    if a.resume:
        if not latest.exists():
            p.error("No latest checkpoint to resume")
        state = torch.load(latest, map_location="cpu", weights_only=False)
        if state["fingerprint"] != fingerprint:
            p.error("Fingerprint mismatch; refusing resume")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        rng.setstate(state["rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        start, best = state["step"], state["best_val"]
        del state
    elif latest.exists():
        p.error("Existing SFT checkpoint. Use --resume.")
    print("SVAROG | STAGE 18F | ADDITION SKILL", flush=True)
    print(f"GPU={torch.cuda.get_device_name(0)} train={len(train)} val={len(validation)} "
          f"seq={a.seq} steps={a.steps} batch={a.batch} accum={a.accum}", flush=True)
    for step in range(start + 1, a.steps + 1):
        tic = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for _ in range(a.accum):
            ids = [rng.randrange(len(train)) for _ in range(a.batch)]
            x, y = make_batch(train, ids, sp.pad_id(), a.seq, "cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_ids=x, labels=y)["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at {step}")
            scaler.scale(loss / a.accum).backward()
            losses.append(float(loss.detach()))
        scaler.unscale_(optimizer)
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad):
            raise RuntimeError(f"Nonfinite gradient at {step}")
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < scale_before:
            raise RuntimeError(f"FP16 overflow; optimizer step skipped at {step}")
        print(f"step={step}/{a.steps} loss={sum(losses)/len(losses):.4f} "
              f"grad={float(grad):.3f} seconds={time.monotonic()-tic:.1f}", flush=True)
        improved = False
        if step % a.eval_every == 0 or step == a.steps:
            val = evaluate(model, validation, sp.pad_id(), a.seq, a.val_batches)
            print(f"EVAL step={step} assistant_loss={val:.5f} ppl={math.exp(min(val,30)):.2f}", flush=True)
            if val < best:
                best, improved = val, True
        if step % a.save_every == 0 or step == a.steps or improved:
            state = {"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scaler": scaler.state_dict(), "rng": rng.getstate(),
                     "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
                     "best_val": best, "fingerprint": fingerprint}
            save_state(latest, state)
            if improved:
                import shutil
                shutil.copy2(latest, best_file)
            if step % a.eval_every == 0 or step == a.steps:
                import shutil
                shutil.copy2(latest, a.output / f"step_{step:04d}.pt")
            print("SAVED", step, "best", best, flush=True)
    print("STAGE 18F COMPLETE")


if __name__ == "__main__":
    main()
