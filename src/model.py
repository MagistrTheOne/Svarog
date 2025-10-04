#!/usr/bin/env python3
"""
Svarog Model Architecture
LLaMA-inspired transformer implementation from scratch
"""

import math
from typing import Optional, Tuple, List
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig


class SvarogConfig(PretrainedConfig):
    """Configuration class for Svarog model"""

    model_type = "svarog"

    def __init__(
        self,
        vocab_size: int = 50000,
        hidden_size: int = 2048,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        intermediate_size: int = 8192,
        hidden_act: str = "silu",
        max_position_embeddings: int = 2048,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        tie_word_embeddings: bool = False,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""

    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Precompute the inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute position-dependent rotations
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)

        return (
            self.cos_cached[:, :, :seq_len, ...],
            self.sin_cached[:, :, :seq_len, ...],
        )

    def apply_rotary_pos_emb(
        self, tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        return (tensor * cos) + (self.rotate_half(tensor) * sin)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embedding to q and k"""
    rope = RotaryPositionEmbedding(q.shape[-1])
    cos, sin = rope(q, q.shape[-2])
    return rope.apply_rotary_pos_emb(q, cos, sin), rope.apply_rotary_pos_emb(k, cos, sin)


class SvarogAttention(nn.Module):
    """Multi-head attention with RoPE"""

    def __init__(self, config: SvarogConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.rope = RotaryPositionEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # Project to q, k, v
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rope(hidden_states, seq_len)
        q = self.rope.apply_rotary_pos_emb(q, cos, sin)
        k = self.rope.apply_rotary_pos_emb(k, cos, sin)

        # Attention computation
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(attn_output)


class SvarogMLP(nn.Module):
    """MLP block with SwiGLU activation"""

    def __init__(self, config: SvarogConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = F.silu  # SwiGLU activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class SvarogDecoderLayer(nn.Module):
    """Transformer decoder layer"""

    def __init__(self, config: SvarogConfig):
        super().__init__()
        self.self_attn = SvarogAttention(config)
        self.mlp = SvarogMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask, position_ids)
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class SvarogModel(PreTrainedModel):
    """Svarog transformer model"""

    config_class = SvarogConfig

    def __init__(self, config: SvarogConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([
            SvarogDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
    ) -> Tuple:
        batch_size, seq_len = input_ids.shape

        # Create position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)

        # Create attention mask
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        all_hidden_states = () if output_hidden_states else None

        # Apply transformer layers
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = torch.utils.checkpoint.checkpoint(layer, hidden_states, attention_mask, position_ids)
            else:
                layer_outputs = layer(hidden_states, attention_mask, position_ids)

            hidden_states = layer_outputs

        # Final layer norm
        hidden_states = self.norm(hidden_states)

        return (hidden_states, all_hidden_states)


class SvarogForCausalLM(PreTrainedModel):
    """Svarog model for causal language modeling"""

    config_class = SvarogConfig

    def __init__(self, config: SvarogConfig):
        super().__init__(config)
        self.model = SvarogModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie weights if specified
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        # Initialize weights
        self.apply(self.model._init_weights)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Tuple:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=False,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift labels for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )

        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> torch.LongTensor:
        """Simple text generation"""
        self.eval()

        for _ in range(max_length - input_ids.shape[1]):
            outputs = self(input_ids=input_ids)
            next_token_logits = outputs["logits"][:, -1, :] / temperature

            if do_sample:
                # Top-p sampling
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_probs[torch.where(cumulative_probs > top_p)] = 0

                # Sample from the remaining tokens
                if sorted_probs.sum() > 0:
                    sorted_probs = sorted_probs / sorted_probs.sum()
                    next_token = torch.multinomial(sorted_probs, 1)
                    next_token = sorted_indices[next_token]
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Stop if EOS token is generated
            if next_token.item() == self.config.eos_token_id:
                break

        return input_ids


def create_svarog_model(config_path: str = "config.json") -> SvarogForCausalLM:
    """Create Svarog model from configuration"""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    config = SvarogConfig(**config_dict["model"])
    model = SvarogForCausalLM(config)
    return model


if __name__ == "__main__":
    # Test model creation
    model = create_svarog_model()
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Test forward pass
    input_ids = torch.randint(0, 50000, (1, 10))
    outputs = model(input_ids=input_ids)
    print(f"Forward pass successful. Loss: {outputs['loss']}, Logits shape: {outputs['logits'].shape}")
