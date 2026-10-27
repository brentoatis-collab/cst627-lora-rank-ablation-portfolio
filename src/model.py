import math
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, dropout: float = 0.0):
        super().__init__()
        self.base = nn.Linear(in_features, out_features)
        self.rank = int(rank)
        self.lora_a = nn.Parameter(torch.empty(self.rank, in_features))
        self.lora_b = nn.Parameter(torch.zeros(out_features, self.rank))
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # alpha == rank, so alpha/rank == 1; capacity is the intended changing factor.
        return self.base(x) + F.linear(F.linear(self.dropout(x), self.lora_a), self.lora_b)


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, rank: int, lora_dropout: float):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model, self.n_heads, self.head_dim = d_model, n_heads, d_model // n_heads
        self.q_proj = LoRALinear(d_model, d_model, rank, lora_dropout)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = LoRALinear(d_model, d_model, rank, lora_dropout)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self._heads(self.q_proj(x)), self._heads(self.k_proj(x)), self._heads(self.v_proj(x))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        weights = self.attn_dropout(scores.softmax(dim=-1))
        values = (weights @ v).transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], self.d_model)
        return self.out_proj(values)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, rank: int, lora_dropout: float):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads, dropout, rank, lora_dropout)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x)))
        return x + self.dropout(self.ff(self.norm2(x)))


class ForecastTransformer(nn.Module):
    def __init__(self, input_features: int, horizon: int, d_model: int, n_heads: int, n_layers: int,
                 d_ff: int, max_sequence_length: int, dropout: float, rank: int, lora_dropout: float = 0.0):
        super().__init__()
        self.input_proj = nn.Linear(input_features, d_model)
        self.position = nn.Parameter(torch.zeros(1, max_sequence_length, d_model))
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff, dropout, rank, lora_dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, horizon)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x) + self.position[:, : x.shape[1]]
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x[:, -1]))


def freeze_for_lora(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for name, parameter in model.named_parameters():
        if ".lora_a" in name or ".lora_b" in name:
            parameter.requires_grad = True


def lora_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    return (parameter for parameter in model.parameters() if parameter.requires_grad)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
