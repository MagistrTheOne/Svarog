"""Svarog smoke test: small random-weight model, no datasets or downloads.

Run from repository root:
    python smoke_2080.py
    python smoke_2080.py --hidden 512 --layers 8 --vocab 8192 --seq 128
This does NOT validate trained checkpoint quality or full-scale training.
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from model import SvarogConfig, SvarogForCausalLM


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--vocab", type=int, default=4096)
    p.add_argument("--seq", type=int, default=64)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--train-step", action="store_true",
                   help="Run a single optimizer step; default tests forward/backward only")
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable. Do not attempt full model.")
    device = torch.device("cuda:0")
    print("GPU:", torch.cuda.get_device_name(0), "Torch:", torch.__version__)
    torch.cuda.reset_peak_memory_stats(device)
    cfg = SvarogConfig(vocab_size=a.vocab, hidden_size=a.hidden,
                       num_hidden_layers=a.layers, num_attention_heads=a.heads,
                       intermediate_size=a.hidden * 4,
                       max_position_embeddings=max(a.seq, 128))
    start = time.perf_counter()
    model = SvarogForCausalLM(cfg).to(device).train()
    params = sum(x.numel() for x in model.parameters())
    print(f"Parameters: {params:,}")
    tokens = torch.randint(3, a.vocab, (a.batch, a.seq), device=device)
    if a.train_step:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        result = model(input_ids=tokens, labels=tokens)
        loss = result["loss"]
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite loss")
    loss.backward()
    if a.train_step:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    print(f"Loss: {loss.item():.5f}")
    print(f"Elapsed: {time.perf_counter()-start:.2f}s")
    print(f"Peak allocated: {torch.cuda.max_memory_allocated(device)/2**20:.1f} MiB")
    print(f"Peak reserved: {torch.cuda.max_memory_reserved(device)/2**20:.1f} MiB")
    print("SMOKE PASS (random weights; not a trained model)")


if __name__ == "__main__":
    main()
