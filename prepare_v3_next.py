"""Prepare a safe NEW-corpus continuation checkpoint for train_v2.py --resume.

Preserves model weights, resets optimizer, data RNG and training fingerprint,
and writes only to the new output directory. Does not modify source checkpoint.
Note: the next train_v2.py run starts from optimizer step 0, while the weights
were pretrained for the source step count.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("config_v3_t4.json"))
    p.add_argument("--train", type=Path, default=Path("data_v3_next/corpus.txt"))
    p.add_argument("--val", type=Path, default=Path("data_v3_next/validation.txt"))
    p.add_argument("--output", type=Path, default=Path("checkpoints/svarog_v3_next"))
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--accum", type=int, default=8)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    dst = a.output / "latest.pt"
    if dst.exists():
        p.error(f"Refusing to overwrite existing continuation checkpoint: {dst}")
    cfg = json.loads(a.config.read_text(encoding="utf-8"))
    tokenizer_path = Path(cfg["tokenizer"]["model_path"])
    fp = {"config": sha(a.config), "tokenizer": sha(tokenizer_path),
          "train": sha(a.train), "val": sha(a.val)}
    state = torch.load(a.source, map_location="cpu", weights_only=False)
    for name in ("config", "tokenizer"):
        if state.get("fingerprints", {}).get(name) != fp[name]:
            p.error(f"Source {name} fingerprint mismatch")
    # Construct the fresh training optimizer state exactly as train_v2.py does.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from model import SvarogConfig, SvarogForCausalLM
    from train_v2 import encode_file
    import sentencepiece as spm
    model = SvarogForCausalLM(SvarogConfig(**cfg["model"]))
    model.load_state_dict(state["model"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg["training"]["learning_rate"],
                                  weight_decay=cfg["training"]["weight_decay"])
    scaler = torch.amp.GradScaler("cuda")
    import random
    seed = int(cfg["training"].get("seed", 42))
    torch.manual_seed(seed)
    random.seed(seed)
    train_rng, val_rng = random.Random(seed), random.Random(seed + 1)
    out = {"step": 0, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
           "scaler": scaler.state_dict(), "fingerprints": fp,
           "seq": a.seq, "batch": a.batch, "accum": a.accum,
           "best_loss": float("inf"), "bad_evals": 0,
           "train_rng": train_rng.getstate(), "val_rng": val_rng.getstate(),
           "torch_rng": torch.get_rng_state(),
           "cuda_rng": torch.cuda.get_rng_state("cuda")}
    # source lineage is metadata only; step 0 is the NEW stage optimizer step.
    out["source_checkpoint"] = str(a.source)
    out["source_step"] = state.get("step")
    tmp = dst.with_suffix(".tmp")
    torch.save(out, tmp)
    tmp.replace(dst)
    print(f"PREPARED: {dst} | base_step={out['source_step']} | continuation_step=0")


if __name__ == "__main__":
    main()
