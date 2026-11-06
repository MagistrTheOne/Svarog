"""Svarog V2 architectural checks: causal isolation, gradients, generation."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from model import SvarogConfig, SvarogForCausalLM


def main():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = SvarogConfig(vocab_size=256, hidden_size=64, num_hidden_layers=2,
                       num_attention_heads=4, intermediate_size=128,
                       max_position_embeddings=32)
    model = SvarogForCausalLM(cfg).to(device).eval()
    x = torch.tensor([[5, 6, 7, 8, 9, 10]], device=device)
    y = x.clone()
    y[0, -1] = 20
    with torch.no_grad():
        first = model(x)["logits"]
        second = model(y)["logits"]
    diff = (first[:, :-1] - second[:, :-1]).abs().max().item()
    print(f"Future-token leakage: {diff:.8f}")
    assert diff < 1e-5, "CAUSAL ATTENTION BROKEN"
    output = model.generate(x[:, :2], max_length=6, do_sample=False)
    assert output.shape == (1, 6)
    print("Generation shape:", tuple(output.shape))
    model.train()
    loss = model(x, labels=x)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    print("Loss:", round(loss.item(), 5))
    print("ARCHITECTURE PASS")


if __name__ == "__main__":
    main()
