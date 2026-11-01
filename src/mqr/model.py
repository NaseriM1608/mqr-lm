"""Shared decoder backbone and interchangeable vocabulary heads for MQR-v1."""

from __future__ import annotations

import math
from typing import TypeAlias

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import ExperimentConfig, ModelConfig
from .heads import build_head
from .retrieval import FaissIPIndex, retrieve_candidates

KVCache: TypeAlias = tuple[tuple[Tensor, Tensor], ...]


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int) -> None:
        super().__init__()
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, values: Tensor, positions: Tensor) -> Tensor:
        angles = torch.outer(positions.float(), self.inv_freq.float())
        cosine, sine = angles.cos().to(values.dtype), angles.sin().to(values.dtype)
        even, odd = values[..., 0::2], values[..., 1::2]
        return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.attention_heads
        self.head_dim = config.d_model // self.num_heads
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(
        self, values: Tensor, positions: Tensor, past: tuple[Tensor, Tensor] | None,
        use_cache: bool,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        batch, length, width = values.shape
        query, key, value = self.qkv(values).view(batch, length, 3, self.num_heads, self.head_dim).unbind(2)
        query, key, value = (item.transpose(1, 2) for item in (query, key, value))
        query, key = self.rope(query, positions), self.rope(key, positions)
        past_length = 0
        if past is not None:
            old_key, old_value = past
            if old_key.device != key.device or old_value.device != value.device:
                raise ValueError("KV cache and input must be on the same device")
            if old_key.dtype != key.dtype or old_value.dtype != value.dtype:
                raise ValueError("KV cache and attention projections must use the same dtype")
            past_length = old_key.size(-2)
            key, value = torch.cat((old_key, key), dim=-2), torch.cat((old_value, value), dim=-2)
        mask = None
        causal = past_length == 0
        if past_length and length > 1:
            key_positions = torch.arange(key.size(-2), device=values.device)
            mask = key_positions.unsqueeze(0) <= positions.unsqueeze(1)
        attended = F.scaled_dot_product_attention(
            query, key, value, attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0, is_causal=causal,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, length, width)
        return self.output(attended), (key, value) if use_cache else None


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.ffn_dim), nn.GELU(),
            nn.Linear(config.ffn_dim, config.d_model),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, values: Tensor, positions: Tensor, past: tuple[Tensor, Tensor] | None,
        use_cache: bool,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        attended, cache = self.attention(self.attention_norm(values), positions, past, use_cache)
        values = values + self.dropout(attended)
        values = values + self.dropout(self.ffn(self.ffn_norm(values)))
        return values, cache


class DecoderBackbone(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(DecoderBlock(config) for _ in range(config.layers))
        self.final_norm = nn.LayerNorm(config.d_model)
        # Initialize the entire backbone before constructing any head. A shared
        # seed therefore gives every head variant exactly the same backbone.
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _cache_length(self, cache: KVCache | None, batch: int) -> int:
        if cache is None:
            return 0
        if not isinstance(cache, (tuple, list)) or len(cache) != len(self.blocks):
            raise ValueError("KV cache must contain one (key, value) pair per decoder layer")
        past_length = None
        for pair in cache:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError("Each KV cache layer must contain a key and a value")
            key, value = pair
            if not isinstance(key, Tensor) or not isinstance(value, Tensor) or key.ndim != 4:
                raise ValueError("KV cache tensors must have shape [batch, heads, length, head_dim]")
            expected = (batch, self.config.attention_heads, key.size(-2), self.config.d_model // self.config.attention_heads)
            if tuple(key.shape) != expected or tuple(value.shape) != expected:
                raise ValueError("KV cache shape does not match this model and input batch")
            if past_length is not None and past_length != key.size(-2):
                raise ValueError("All KV cache layers must have the same cached length")
            past_length = key.size(-2)
        return int(past_length or 0)

    def forward(
        self, input_ids: Tensor, past_key_values: KVCache | None = None, use_cache: bool = False,
    ) -> tuple[Tensor, KVCache | None]:
        if input_ids.ndim != 2 or input_ids.size(0) == 0 or input_ids.size(1) == 0:
            raise ValueError("input_ids must have nonempty shape [batch, sequence]")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must contain integer token IDs")
        past_length = self._cache_length(past_key_values, input_ids.size(0))
        total_length = past_length + input_ids.size(1)
        if total_length > self.config.context_length:
            raise ValueError(f"Input and cache length {total_length} exceed context_length={self.config.context_length}")
        positions = torch.arange(past_length, total_length, device=input_ids.device)
        values = self.dropout(self.embedding(input_ids))
        new_cache = []
        for index, block in enumerate(self.blocks):
            old_cache = None if past_key_values is None else past_key_values[index]
            values, cache = block(values, positions, old_cache, use_cache)
            if cache is not None:
                new_cache.append(cache)
        return self.final_norm(values), tuple(new_cache) if use_cache else None


class LanguageModel(nn.Module):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.backbone = DecoderBackbone(config.model)
        self.head = build_head(config.head, config.model.d_model, config.model.vocab_size)

    def forward(self, input_ids: Tensor) -> Tensor:
        features, _ = self.backbone(input_ids)
        return self.head.score_all(features)

    @torch.no_grad()
    def generate(
        self, input_ids: Tensor, max_new_tokens: int, temperature: float = 1.0,
        top_k: int | None = None, eos_token_id: int | None = None,
        *, method: str = "exact", k: int = 32,
    ) -> Tensor:
        """Generate with a KV cache and exact or retrieved token scoring.

        Retrieval ``k`` is per query; ``top_k`` limits sampling within the union.
        Every candidate is reranked with all queries. Retrieval sampling uses a
        truncated distribution and can differ from exact generation. Faiss CPU
        indexes are built once per call, and CUDA models include CPU transfers.
        EOS is included as an ordinary scored candidate when supplied.
        """
        if type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a nonnegative integer")
        if input_ids.ndim != 2 or input_ids.size(0) == 0 or input_ids.size(1) == 0:
            raise ValueError("input_ids must have nonempty shape [batch, sequence]")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative; zero selects greedy decoding")
        if top_k is not None and (type(top_k) is not int or not 1 <= top_k <= self.config.model.vocab_size):
            raise ValueError("top_k must be between 1 and vocab_size")
        if eos_token_id is not None and (type(eos_token_id) is not int or not 0 <= eos_token_id < self.config.model.vocab_size):
            raise ValueError("eos_token_id must be a valid vocabulary ID")
        if method not in ("exact", "flat", "faiss", "hnsw"):
            raise ValueError("method must be exact, flat, faiss, or hnsw")
        if type(k) is not int or k < 1:
            raise ValueError("k must be a positive integer")
        if input_ids.size(1) + max_new_tokens > self.config.model.context_length:
            raise ValueError("Requested generation exceeds model context_length")
        if max_new_tokens == 0:
            return input_ids.clone()
        was_training = self.training
        self.eval()
        tokens, cache = input_ids, None
        current = input_ids
        finished = torch.zeros(input_ids.size(0), device=input_ids.device, dtype=torch.bool)
        try:
            index = None
            if method in ("faiss", "hnsw"):
                index = FaissIPIndex(self.head, "flat" if method == "faiss" else "hnsw")
            always_include = () if eos_token_id is None else (eos_token_id,)
            for _ in range(max_new_tokens):
                features, cache = self.backbone(current, past_key_values=cache, use_cache=True)
                candidate_ids = None
                if method == "exact":
                    logits = self.head.score_all(features[:, -1]).float()
                else:
                    candidates = retrieve_candidates(self.head, features[:, -1], k, method, index, always_include)
                    candidate_ids = candidates.ids
                    logits = candidates.scores.float()
                if temperature == 0:
                    next_token = logits.argmax(dim=-1, keepdim=True)
                else:
                    logits = logits / temperature
                    if top_k is not None:
                        values, indices = logits.topk(min(top_k, logits.size(-1)), dim=-1)
                        sampled = torch.multinomial(F.softmax(values, dim=-1), 1)
                        next_token = indices.gather(-1, sampled)
                    else:
                        next_token = torch.multinomial(F.softmax(logits, dim=-1), 1)
                if candidate_ids is not None:
                    next_token = candidate_ids.gather(-1, next_token)
                if eos_token_id is not None:
                    next_token = torch.where(finished[:, None], eos_token_id, next_token)
                    finished |= next_token.squeeze(-1) == eos_token_id
                tokens = torch.cat((tokens, next_token), dim=1)
                if eos_token_id is not None and finished.all():
                    break
                current = next_token
        finally:
            self.train(was_training)
        return tokens
