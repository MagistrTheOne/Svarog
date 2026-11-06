"""Native PyTorch pretraining for Svarog V2 on an 8GB CUDA GPU.

Examples:
  python train_v2.py --steps 20 --seq 128 --eval-every 10 --save-every 10
  python train_v2.py --steps 2000 --seq 128 --resume

No DeepSpeed, downloads or global dependency changes.
"""
import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from model import SvarogConfig, SvarogForCausalLM


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def encode_file(path, tokenizer, max_docs=None):
    if not path.is_file():
        raise FileNotFoundError(path)
    documents = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids = [tokenizer.bos_id()] + tokenizer.encode(line, out_type=int) + [tokenizer.eos_id()]
            if len(ids) >= 3:
                documents.append(ids)
            if max_docs and len(documents) >= max_docs:
                break
    if not documents:
        raise RuntimeError(f"No usable documents in {path}")
    return documents


def make_batch(docs, seq, batch, rng, device, pad_id):
    """Length-weighted document sampling; prefer full windows over EOS tails.

    A document is never mixed with another one inside a sample. Short
    documents are supported but not preferentially oversampled.
    """
    x = torch.full((batch, seq), pad_id, dtype=torch.long)
    y = torch.full((batch, seq), -100, dtype=torch.long)
    for i in range(batch):
        doc = docs[rng.randrange(len(docs))]
        if len(doc) > seq:
            start = rng.randrange(len(doc) - seq + 1)
            chunk = doc[start:start + seq]
        else:
            chunk = doc
        n = len(chunk)
        x[i, :n] = torch.tensor(chunk)
        y[i, :n] = torch.tensor(chunk)
    return x.to(device), y.to(device)


def evaluate(model, docs, seq, batch, rng, device, pad_id, batches):
    model.eval()
    losses = []
    with torch.inference_mode():
        for _ in range(batches):
            x, y = make_batch(docs, seq, batch, rng, device, pad_id)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(x, labels=y)["loss"]
            if torch.isfinite(loss):
                losses.append(float(loss))
    model.train()
    if not losses:
        raise RuntimeError("Validation returned no finite losses")
    return sum(losses) / len(losses)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("config_v2_2080.json"))
    p.add_argument("--train", type=Path, default=Path("data/corpus.txt"))
    p.add_argument("--val", type=Path, default=Path("data/validation.txt"))
    p.add_argument("--output", type=Path, default=Path("checkpoints/svarog_v2_2080"))
    p.add_argument("--steps", type=int, default=2000, help="Total optimizer steps, including resumed steps")
    p.add_argument("--seq", type=int, default=128)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--val-batches", type=int, default=8)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--early-stop-patience", type=int, default=8, help="Consecutive evaluations without improvement; 0 disables")
    a = p.parse_args()
    if not torch.cuda.is_available():
        p.error("CUDA is required for this training entrypoint")
    if min(a.steps, a.seq, a.batch, a.accum, a.eval_every, a.save_every, a.val_batches) < 1 or a.early_stop_patience < 0:
        p.error("All numeric settings must be positive")
    config = json.loads(a.config.read_text(encoding="utf-8"))
    cfg = SvarogConfig(**config["model"])
    if a.seq > cfg.max_position_embeddings:
        p.error("seq exceeds model context")
    tokenizer_path = Path(config["tokenizer"]["model_path"])
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    if tokenizer.vocab_size() != cfg.vocab_size:
        p.error("Tokenizer vocabulary and model vocabulary differ")
    for name in ("pad", "bos", "eos"):
        if getattr(tokenizer, f"{name}_id")() != getattr(cfg, f"{name}_token_id"):
            p.error(f"{name} token ID mismatch")
    train_docs = encode_file(a.train, tokenizer)
    val_docs = encode_file(a.val, tokenizer)
    print(f"Documents: train={len(train_docs)} val={len(val_docs)}")
    if len(train_docs) < 10 or len(val_docs) < 2:
        p.error("Corpus too small: require >=10 train docs and >=2 validation docs")
    device = torch.device("cuda")
    seed = int(config["training"].get("seed", 42))
    torch.manual_seed(seed)
    random.seed(seed)
    train_rng, val_rng = random.Random(seed), random.Random(seed + 1)
    # Fixed validation windows across evaluations for comparable loss.
    validation_seed_state = val_rng.getstate()
    best_loss = float("inf")
    bad_evals = 0
    model = SvarogForCausalLM(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"],
                                  weight_decay=config["training"]["weight_decay"])
    scaler = torch.amp.GradScaler("cuda")
    a.output.mkdir(parents=True, exist_ok=True)
    ckpt = a.output / "latest.pt"
    step_start = 0
    fingerprints = {"config": sha256(a.config), "tokenizer": sha256(tokenizer_path),
                    "train": sha256(a.train), "val": sha256(a.val)}
    if a.resume:
        if not ckpt.exists():
            p.error(f"Missing checkpoint: {ckpt}")
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        if state["fingerprints"] != fingerprints:
            p.error("Checkpoint/data/tokenizer/config mismatch; refusing unsafe resume")
        if state["seq"] != a.seq or state["batch"] != a.batch or state["accum"] != a.accum:
            p.error("Resume requires same seq, batch and accum")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        train_rng.setstate(state["train_rng"])
        val_rng.setstate(state["val_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"], device)
        step_start = state["step"]
        best_loss = state.get("best_loss", float("inf"))
        bad_evals = state.get("bad_evals", 0)
        print("Resumed optimizer step", step_start)
    print("GPU:", torch.cuda.get_device_name(0),
          "parameters:", f"{sum(p.numel() for p in model.parameters()):,}")
    print("Training with real text; initial loss is NOT a quality benchmark.")
    model.train()
    for step in range(step_start + 1, a.steps + 1):
        improved = False
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for _ in range(a.accum):
            x, y = make_batch(train_docs, a.seq, a.batch, train_rng, device, tokenizer.pad_id())
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(x, labels=y)["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at step {step}")
            scaler.scale(loss / a.accum).backward()
            losses.append(float(loss.detach()))
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["max_grad_norm"])
        scaler.step(optimizer)
        scaler.update()
        print(f"step={step}/{a.steps} train_loss={sum(losses)/len(losses):.4f} "
              f"vram_peak_MiB={torch.cuda.max_memory_allocated()/2**20:.1f}", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            val_rng.setstate(validation_seed_state)
            val_loss = evaluate(model, val_docs, a.seq, a.batch, val_rng, device,
                                tokenizer.pad_id(), a.val_batches)
            print(f"step={step} val_loss={val_loss:.4f} val_ppl={math.exp(min(val_loss, 30)):.2f}", flush=True)
            improved = val_loss < best_loss
            if improved:
                best_loss, bad_evals = val_loss, 0
            else:
                bad_evals += 1
        stop = bool(a.early_stop_patience and bad_evals >= a.early_stop_patience)
        if step % a.save_every == 0 or step == a.steps or improved or stop:
            state = {
                "step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(), "fingerprints": fingerprints,
                "seq": a.seq, "batch": a.batch, "accum": a.accum,
                "best_loss": best_loss, "bad_evals": bad_evals,
                "train_rng": train_rng.getstate(), "val_rng": val_rng.getstate(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device),
            }
            temp = ckpt.with_suffix(".tmp")
            torch.save(state, temp)
            temp.replace(ckpt)
            print("Saved:", ckpt, flush=True)
            if improved:
                best = a.output / "best.pt"
                import shutil
                shutil.copy2(ckpt, best)
                print("Best validation checkpoint:", best, flush=True)
        if stop:
            print("EARLY STOP: validation did not improve", a.early_stop_patience, "times")
            break
    print("TRAINING COMPLETE")


if __name__ == "__main__":
    main()
