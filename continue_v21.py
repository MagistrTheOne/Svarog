"""Continue Svarog language pretraining on a NEW corpus without overwriting V2.

Run from repo root:
  python continue_v21.py --source checkpoints/svarog_v2_2080/best.pt \
    --train data_v21/corpus.txt --val data_v21/validation.txt \
    --output /content/drive/MyDrive/NULLXES/SVAROG/V21 --steps 1000

This intentionally resets optimizer state and saves model-only checkpoints.
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


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def read_docs(path, sp):
    docs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids = [sp.bos_id()] + sp.encode(line, out_type=int) + [sp.eos_id()]
                if len(ids) >= 8:
                    docs.append(ids)
    if not docs:
        raise ValueError(f"No usable text: {path}")
    return docs


def draw(docs, seq, batch, rng, pad, device):
    x = torch.full((batch, seq), pad, dtype=torch.long)
    y = torch.full((batch, seq), -100, dtype=torch.long)
    for i in range(batch):
        doc = docs[rng.randrange(len(docs))]
        start = rng.randrange(len(doc) - seq + 1) if len(doc) > seq else 0
        chunk = doc[start:start + seq]
        n = len(chunk)
        x[i, :n] = torch.tensor(chunk)
        y[i, :n] = torch.tensor(chunk)
    return x.to(device), y.to(device)


def evaluate(model, docs, seq, rng_seed, pad, device, batches):
    rng = random.Random(rng_seed)
    model.eval()
    total = 0.
    with torch.inference_mode():
        for _ in range(batches):
            x, y = draw(docs, seq, 1, rng, pad, device)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x)["logits"]
            loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)),
                                   y[:, 1:].reshape(-1), ignore_index=-100)
            total += loss.item()
    model.train()
    return total / batches


def save(path, model, step, fp, loss, source):
    state = {"model": model.state_dict(), "step": step, "fingerprints": fp,
             "validation_loss": loss, "source_checkpoint": str(source)}
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("config_v2_2080.json"))
    p.add_argument("--source", type=Path, default=Path("checkpoints/svarog_v2_2080/best.pt"))
    p.add_argument("--train", type=Path, default=Path("data_v21/corpus.txt"))
    p.add_argument("--val", type=Path, default=Path("data_v21/validation.txt"))
    p.add_argument("--output", type=Path, default=Path("checkpoints/svarog_v21"))
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.00005)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--val-batches", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if not torch.cuda.is_available():
        p.error("CUDA GPU required")
    if min(a.steps, a.seq, a.batch, a.accum, a.eval_every, a.val_batches) < 1 or a.lr <= 0:
        p.error("Invalid hyperparameters")
    cfg_json = json.loads(a.config.read_text(encoding="utf-8"))
    cfg = SvarogConfig(**cfg_json["model"])
    if a.seq > cfg.max_position_embeddings:
        p.error("Sequence exceeds model context")
    tok_path = Path(cfg_json["tokenizer"]["model_path"])
    sp = spm.SentencePieceProcessor(model_file=str(tok_path))
    if sp.vocab_size() != cfg.vocab_size:
        p.error("Tokenizer vocabulary mismatch")
    fp = {"config": sha(a.config), "tokenizer": sha(tok_path),
          "train": sha(a.train), "val": sha(a.val)}
    base = torch.load(a.source, map_location="cpu", weights_only=False)
    for k in ("config", "tokenizer"):
        if base.get("fingerprints", {}).get(k) != fp[k]:
            p.error(f"Source checkpoint {k} mismatch")
    model = SvarogForCausalLM(cfg)
    model.load_state_dict(base["model"], strict=True)
    del base
    model = model.cuda().train()
    train, val = read_docs(a.train, sp), read_docs(a.val, sp)
    if len(train) < 100 or len(val) < 10:
        p.error("Require >=100 training and >=10 validation documents")
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    scaler = torch.amp.GradScaler("cuda")
    a.output.mkdir(parents=True, exist_ok=True)
    best_loss = math.inf
    print(f"SVAROG V2.1 | GPU={torch.cuda.get_device_name(0)} | "
          f"train_docs={len(train)} val_docs={len(val)} | source={a.source}", flush=True)
    for step in range(1, a.steps + 1):
        opt.zero_grad(set_to_none=True)
        running = 0.
        for _ in range(a.accum):
            x, y = draw(train, a.seq, a.batch, rng, sp.pad_id(), "cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x)["logits"]
            loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, cfg.vocab_size),
                                   y[:, 1:].reshape(-1), ignore_index=-100)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at step {step}")
            scaler.scale(loss / a.accum).backward()
            running += loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        scaler.step(opt)
        scaler.update()
        if step % a.eval_every == 0 or step == a.steps:
            v = evaluate(model, val, a.seq, a.seed + 9000, sp.pad_id(), "cuda", a.val_batches)
            print(f"step={step}/{a.steps} train_loss={running/a.accum:.4f} "
                  f"val_loss={v:.4f} val_ppl={math.exp(min(v,30)):.1f}", flush=True)
            save(a.output / "latest.pt", model, step, fp, v, a.source)
            if v < best_loss:
                best_loss = v
                save(a.output / "best.pt", model, step, fp, v, a.source)
                print("Saved improved best.pt", flush=True)
    print("V2.1 COMPLETE | best_val_loss=", best_loss)


if __name__ == "__main__":
    main()
