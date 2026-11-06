"""SVAROG identity SFT: train answers only, preserve the base checkpoint.

python identity_sft_v2.py --steps 150
python infer_v2.py --checkpoint checkpoints/svarog_identity_v1/best.pt --identity-chat --prompt "Кто ты?"
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


def load_examples(path, sp, limit):
    examples = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            q, a = obj["question"].strip(), obj["answer"].strip()
            if not q or not a:
                raise ValueError("Empty question or answer")
            prefix = sp.encode("Вопрос: " + q + "\nОтвет:", out_type=int)
            answer = sp.encode(" " + a, out_type=int) + [sp.eos_id()]
            ids = [sp.bos_id()] + prefix + answer
            if len(ids) > limit:
                raise ValueError(f"Example exceeds {limit} tokens: {q}")
            # Loss on answer only. LM.forward shifts logits/labels by 1.
            labels = [-100] * (1 + len(prefix)) + answer
            examples.append((ids, labels))
    if len(examples) < 12:
        raise ValueError("Need at least 12 identity examples")
    return examples


def batch_for(examples, indices, pad, device):
    length = max(len(examples[i][0]) for i in indices)
    x = torch.full((len(indices), length), pad, dtype=torch.long)
    y = torch.full((len(indices), length), -100, dtype=torch.long)
    for row, i in enumerate(indices):
        ids, labels = examples[i]
        x[row, :len(ids)] = torch.tensor(ids)
        y[row, :len(labels)] = torch.tensor(labels)
    return x.to(device), y.to(device)


def eval_loss(model, examples, pad, device):
    model.eval()
    losses = []
    with torch.inference_mode():
        for i in range(len(examples)):
            x, y = batch_for(examples, [i], pad, device)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x)["logits"]
                loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)),
                                       y[:, 1:].reshape(-1), ignore_index=-100)
            losses.append(float(loss))
    model.train()
    return sum(losses) / len(losses)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("config_v2_2080.json"))
    p.add_argument("--source", type=Path, default=Path("checkpoints/svarog_v2_2080/best.pt"))
    p.add_argument("--data", type=Path, default=Path("data_identity/svarog_identity_v1.jsonl"))
    p.add_argument("--output", type=Path, default=Path("checkpoints/svarog_identity_v1"))
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.00002)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if not torch.cuda.is_available():
        p.error("CUDA required")
    if min(a.steps, a.batch, a.accum, a.eval_every, a.patience) < 1 or a.lr <= 0:
        p.error("Invalid training arguments")
    cfg_data = json.loads(a.config.read_text(encoding="utf-8"))
    cfg = SvarogConfig(**cfg_data["model"])
    sp_path = Path(cfg_data["tokenizer"]["model_path"])
    sp = spm.SentencePieceProcessor(model_file=str(sp_path))
    if sp.vocab_size() != cfg.vocab_size:
        p.error("Tokenizer vocabulary mismatch")
    records = load_examples(a.data, sp, cfg.max_position_embeddings)
    # Fixed held-out examples. They are not used for gradient updates.
    rng = random.Random(a.seed)
    rng.shuffle(records)
    n_val = max(4, len(records) // 10)
    val, train = records[:n_val], records[n_val:]
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    model = SvarogForCausalLM(cfg).to("cuda")
    # The original checkpoint was produced locally. Do not load untrusted .pt.
    base = torch.load(a.source, map_location="cpu", weights_only=False)
    fp = base.get("fingerprints", {})
    if fp.get("config") != sha(a.config) or fp.get("tokenizer") != sha(sp_path):
        p.error("Base checkpoint configuration or tokenizer mismatch")
    model.load_state_dict(base["model"], strict=True)
    del base
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    a.output.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    stale = 0
    print(f"SVAROG IDENTITY SFT | train={len(train)} validation={len(val)} | "
          f"source={a.source} | output={a.output}", flush=True)
    model.train()
    for step in range(1, a.steps + 1):
        opt.zero_grad(set_to_none=True)
        running = 0.0
        for _ in range(a.accum):
            chosen = [rng.randrange(len(train)) for _ in range(a.batch)]
            x, y = batch_for(train, chosen, sp.pad_id(), "cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x)["logits"]
                # Compute CE in fp32 to reduce overflow.
                loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, cfg.vocab_size),
                                       y[:, 1:].reshape(-1), ignore_index=-100)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite SFT loss at step {step}")
            scaler.scale(loss / a.accum).backward()
            running += float(loss.detach())
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if step % a.eval_every == 0 or step == a.steps:
            validation = eval_loss(model, val, sp.pad_id(), "cuda")
            print(f"step={step}/{a.steps} train_loss={running/a.accum:.4f} "
                  f"val_loss={validation:.4f}", flush=True)
            if validation < best:
                best = validation
                stale = 0
                # Store only weights and small metadata, not optimizer state.
                state = {"model": model.state_dict(), "step": step,
                         "fingerprints": fp, "identity_data_sha256": sha(a.data),
                         "source_checkpoint": str(a.source), "validation_loss": best}
                path = a.output / "best.pt"
                temp = a.output / "best.tmp"
                torch.save(state, temp)
                temp.replace(path)
                print("Saved best:", path, flush=True)
            else:
                stale += 1
                if stale >= a.patience:
                    print("Early stop: validation plateau", flush=True)
                    break
    print("SFT COMPLETE | best_val_loss=", round(best, 4))


if __name__ == "__main__":
    main()
