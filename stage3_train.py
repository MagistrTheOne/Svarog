#!/usr/bin/env python3
"""SVAROG Stage 13: full-parameter Windows RTX 2080 SUPER continued pretraining.

Run from repository root. Loads HF release weights; never overwrites hf_model.
Builds compact memory-mapped uint32 token caches from Stage 3 train/validation.
Correct causal labels: model(input_ids=x, labels=y) with y=x except padding -100.
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

import numpy as np
import torch


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare_cache(path, cache, sp, tokenizer_sha):
    """Build durable compact cache; verify input fingerprints on every run."""
    cache.mkdir(parents=True, exist_ok=True)
    fingerprint = sha256(path)
    meta_path = cache / "meta.json"
    bin_path = cache / "tokens.bin"
    idx_path = cache / "offsets.npy"
    if meta_path.exists() and bin_path.exists() and idx_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["text_sha256"] == fingerprint and meta["tokenizer_sha256"] == tokenizer_sha:
            offsets = np.load(idx_path, mmap_mode="r")
            if bin_path.stat().st_size == int(offsets[-1]) * 4:
                print("CACHE REUSED:", path, f"tokens={int(offsets[-1]):,}", flush=True)
                return meta
        raise RuntimeError(f"Cache mismatch: {cache}; do not overwrite silently")
    if any(x.exists() for x in (meta_path, bin_path, idx_path)):
        raise RuntimeError(f"Incomplete token cache in {cache}. Inspect before deleting.")
    tmp_bin = cache / "tokens.bin.partial"
    tmp_idx = cache / "offsets.npy.partial"
    tmp_meta = cache / "meta.json.partial"
    offsets = [0]
    docs = 0
    with path.open("r", encoding="utf-8") as f, tmp_bin.open("wb") as out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids = [sp.bos_id()] + sp.encode(line, out_type=int) + [sp.eos_id()]
            np.asarray(ids, dtype=np.uint32).tofile(out)
            offsets.append(offsets[-1] + len(ids))
            docs += 1
            if docs % 10000 == 0:
                print(f"TOKENIZE {path.name}: {docs:,} docs / {offsets[-1]:,} tokens", flush=True)
        out.flush()
        os.fsync(out.fileno())
    if docs < 10:
        raise RuntimeError(f"Too few documents in {path}")
    with tmp_idx.open("wb") as out:
        np.save(out, np.asarray(offsets, dtype=np.int64))
    meta = {"text_sha256": fingerprint, "tokenizer_sha256": tokenizer_sha,
            "docs": docs, "tokens": offsets[-1]}
    tmp_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    os.replace(tmp_bin, bin_path)
    os.replace(tmp_idx, idx_path)
    os.replace(tmp_meta, meta_path)
    print(f"CACHE READY: {path.name} docs={docs:,} tokens={offsets[-1]:,}", flush=True)
    return meta


class TokenCorpus:
    def __init__(self, folder):
        self.offsets = np.load(folder / "offsets.npy", mmap_mode="r")
        self.tokens = np.memmap(folder / "tokens.bin", dtype=np.uint32, mode="r")
        self.docs = len(self.offsets) - 1

    def sample(self, seq, rng, pad):
        for _ in range(50):
            i = rng.randrange(self.docs)
            start, end = int(self.offsets[i]), int(self.offsets[i + 1])
            length = end - start
            if length >= 16:
                break
        else:
            raise RuntimeError("No sufficiently long documents")
        if length > seq:
            start += rng.randrange(length - seq + 1)
            length = seq
        n = min(seq, length)
        x = np.full(seq, pad, dtype=np.int64)
        y = np.full(seq, -100, dtype=np.int64)
        x[:n] = self.tokens[start:start + n]
        y[:n] = x[:n]
        return x, y


def batch(corpus, seq, micro, rng, pad, device):
    samples = [corpus.sample(seq, rng, pad) for _ in range(micro)]
    x = torch.from_numpy(np.stack([s[0] for s in samples])).to(device)
    y = torch.from_numpy(np.stack([s[1] for s in samples])).to(device)
    return x, y


def evaluate(model, corpus, seq, micro, pad, batches, seed):
    model.eval()
    rng = random.Random(seed)
    total = 0.0
    with torch.inference_mode():
        for _ in range(batches):
            x, y = batch(corpus, seq, micro, rng, pad, "cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_ids=x, labels=y)["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite validation loss")
            total += float(loss)
    model.train()
    return total / batches


def save_state(path, state):
    temp = path.with_suffix(".tmp")
    torch.save(state, temp)
    os.replace(temp, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, default=Path("hf_model"))
    p.add_argument("--data-dir", type=Path, default=Path("data_stage3_50m"))
    p.add_argument("--output", type=Path, default=Path("stage3/checkpoints"))
    p.add_argument("--steps", type=int, default=6000, help="Total optimizer steps, including resume")
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--accum", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--val-batches", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    if not torch.cuda.is_available():
        p.error("CUDA required")
    if min(a.steps, a.seq, a.batch, a.accum, a.eval_every, a.save_every, a.val_batches) < 1 or a.lr <= 0:
        p.error("Invalid training arguments")
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "cache_train").mkdir(exist_ok=True)
    (a.output / "cache_validation").mkdir(exist_ok=True)
    sys.path.insert(0, str(a.model_dir.resolve()))
    from inference import load_svarog
    import sentencepiece as spm

    tokenizer_file = a.model_dir / "tokenizer.model"
    weights_file = a.model_dir / "model.safetensors"
    if not tokenizer_file.is_file() or not weights_file.is_file():
        p.error("Missing HF model weights or tokenizer")
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_file))
    if sp.vocab_size() != 24000:
        p.error("Wrong tokenizer vocabulary")
    tok_sha = sha256(tokenizer_file)
    train_path, val_path = a.data_dir / "corpus.txt", a.data_dir / "validation.txt"
    for path in (train_path, val_path):
        if not path.is_file():
            p.error(f"Missing Stage 3 dataset: {path}")
    train_meta = prepare_cache(train_path, a.output / "cache_train", sp, tok_sha)
    val_meta = prepare_cache(val_path, a.output / "cache_validation", sp, tok_sha)
    train = TokenCorpus(a.output / "cache_train")
    validation = TokenCorpus(a.output / "cache_validation")

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    rng = random.Random(a.seed)
    model, _ = load_svarog(a.model_dir.resolve(), device="cpu")
    if a.seq > model.config.max_position_embeddings:
        p.error("Sequence exceeds model context capacity")
    model = model.to("cuda").train()
    if sum(x.numel() for x in model.parameters()) != 131697408:
        p.error("Unexpected parameter count")
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    scaler = torch.amp.GradScaler("cuda")
    latest = a.output / "latest.pt"
    best_file = a.output / "best.pt"
    fingerprint = {
        "weights_sha256": sha256(weights_file),
        "tokenizer_sha256": tok_sha,
        "train_sha256": train_meta["text_sha256"],
        "validation_sha256": val_meta["text_sha256"],
        "seq": a.seq, "batch": a.batch, "accum": a.accum,
        "lr": a.lr, "weight_decay": a.weight_decay, "seed": a.seed,
    }
    start, best = 0, float("inf")
    if a.resume:
        if not latest.exists():
            p.error(f"Cannot resume; missing {latest}")
        state = torch.load(latest, map_location="cpu", weights_only=False)
        if state["fingerprint"] != fingerprint:
            p.error("Resume fingerprint mismatch; refusing unsafe resume")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        rng.setstate(state["train_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        start, best = int(state["step"]), float(state["best_val"])
        del state
        print(f"RESUME step={start}", flush=True)
    elif latest.exists():
        p.error("Existing checkpoint detected. Use --resume or choose a new --output.")
    print("=" * 70)
    print("NULLXES SVAROG | STAGE 13 | FULL PARAMETER CONTINUED PRETRAINING")
    print("GPU:", torch.cuda.get_device_name(0))
    print(f"train={train_meta['tokens']:,} val={val_meta['tokens']:,} tokens")
    print(f"steps={a.steps} seq={a.seq} batch={a.batch} accum={a.accum} lr={a.lr}")
    print("BASE: HF model.safetensors | TOKENIZER: unchanged | OPTIMIZER: AdamW")
    print("=" * 70, flush=True)
    model.train()
    for step in range(start + 1, a.steps + 1):
        tic = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        for _ in range(a.accum):
            x, y = batch(train, a.seq, a.batch, rng, sp.pad_id(), "cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_ids=x, labels=y)["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite training loss at step {step}")
            scaler.scale(loss / a.accum).backward()
            train_loss += float(loss.detach())
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"Nonfinite gradient norm at step {step}")
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < scale_before:
            raise RuntimeError(f"FP16 overflow: optimizer step skipped at {step}")
        print(f"step={step}/{a.steps} train_loss={train_loss/a.accum:.4f} "
              f"grad={float(grad_norm):.3f} peak_vram_GiB={torch.cuda.max_memory_allocated()/2**30:.2f} "
              f"seconds={time.monotonic()-tic:.1f}", flush=True)
        improved = False
        if step % a.eval_every == 0 or step == a.steps:
            val_loss = evaluate(model, validation, a.seq, a.batch, sp.pad_id(), a.val_batches, a.seed + 2000)
            print(f"EVAL step={step} loss={val_loss:.5f} ppl={math.exp(min(val_loss,30)):.2f}", flush=True)
            if val_loss < best:
                best, improved = val_loss, True
        if step % a.save_every == 0 or step == a.steps or improved:
            state = {
                "step": step, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                "train_rng": rng.getstate(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(),
                "best_val": best, "fingerprint": fingerprint
            }
            save_state(latest, state)
            if improved:
                import shutil
                shutil.copy2(latest, best_file)
            print(f"SAVED step={step} best_val={best:.5f}", flush=True)
    print("STAGE 13 COMPLETE")


if __name__ == "__main__":
    main()
