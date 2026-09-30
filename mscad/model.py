"""Multi-scale reconstruction network with synchronous cross-scale attention."""

from numbers import Integral, Real
from typing import Sequence, Tuple

import math
import torch
from torch import nn
from torch.nn import functional as F


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _patch_sizes(values: Sequence[int]) -> Tuple[int, ...]:
    try:
        patches = tuple(values)
    except TypeError as exc:
        raise ValueError("patch_sizes must be a nonempty sequence of even integers.") from exc
    if not patches:
        raise ValueError("patch_sizes must be nonempty.")
    for patch in patches:
        _positive_int("Each patch size", patch)
        if patch < 2 or patch % 2:
            raise ValueError("Patch sizes must be even integers >= 2 for half-patch strides.")
    if len(set(patches)) != len(patches):
        raise ValueError("patch_sizes must not contain duplicates.")
    return tuple(int(patch) for patch in patches)


def _dropout(value: float) -> float:
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or not 0 <= value < 1):
        raise ValueError("dropout must be finite and in [0, 1).")
    return float(value)


class PatchEmbedding(nn.Module):
    """Project overlapping univariate patches, then normalize token features."""

    def __init__(self, patch_size: int, d_model: int):
        super().__init__()
        self.patch_size = patch_size
        self.stride = patch_size // 2
        self.projection = nn.Linear(patch_size, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = x.squeeze(-1).unfold(1, self.patch_size, self.stride)
        return self.norm(self.projection(patches))


class LearnablePositionalEncoding(nn.Module):
    """Learnable positions; keep the surviving implementation's 512-token table."""

    def __init__(self, d_model: int):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(1):
            raise ValueError("A scale has more than 512 tokens; reduce win_size or increase patch sizes.")
        return x + self.pe[:, :x.size(1)]


class ScaleBranch(nn.Module):
    """One patch scale: embedding, Transformer encoder, reconstruction decoder."""

    def __init__(self, patch_size: int, d_model: int, n_heads: int,
                 n_layers: int, dropout: float):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = PatchEmbedding(patch_size, d_model)
        self.pos_enc = LearnablePositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.decoder = nn.Linear(d_model, patch_size)

    def encode(self, x: torch.Tensor):
        encoded = self.encoder(self.pos_enc(self.patch_embed(x)))
        original = x.squeeze(-1).unfold(1, self.patch_size, self.patch_embed.stride)
        return encoded, original

    def decode(self, encoded: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        reconstructed = self.decoder(encoded)
        # Every patch contributes equally to the scalar error for this scale.
        return F.mse_loss(reconstructed, original, reduction="none").mean(dim=-1).mean(dim=-1)


class CrossScaleBridge(nn.Module):
    """One symmetric exchange block with attention and FFN shared across scales.

    Every query reads the input list. Outputs are collected separately so a
    later scale cannot observe an earlier scale's update within this block.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout),
        )

    def forward(self, encoded_list):
        if len(encoded_list) < 2:
            return encoded_list
        outputs = []
        for index, query in enumerate(encoded_list):
            context = torch.cat([tokens for other, tokens in enumerate(encoded_list)
                                 if other != index], dim=1)
            attention, _ = self.cross_attn(query, context, context, need_weights=False)
            residual = self.norm1(query + attention)
            outputs.append(self.norm2(residual + self.ff(residual)))
        return outputs


class MSCADModel(nn.Module):
    """Score single-channel windows of shape ``(batch, time, 1)``.

    Returns ``(scale_errors, window_scores)`` with shapes ``(batch, scales)``
    and ``(batch,)``. Window scores are uniform means of reconstruction MSEs.
    One instance is shared by every channel in :class:`mscad.MSCAD`.
    """

    def __init__(self, patch_sizes: Sequence[int] = (4, 16, 64), d_model: int = 256,
                 n_heads: int = 4, n_layers_per_scale: int = 2,
                 bridge_depth: int = 2, dropout: float = 0.1):
        super().__init__()
        self.patch_sizes = _patch_sizes(patch_sizes)
        d_model = _positive_int("d_model", d_model)
        n_heads = _positive_int("n_heads", n_heads)
        n_layers_per_scale = _positive_int("n_layers_per_scale", n_layers_per_scale)
        bridge_depth = _positive_int("bridge_depth", bridge_depth)
        dropout = _dropout(dropout)
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads.")
        self.branches = nn.ModuleList([
            ScaleBranch(patch, d_model, n_heads, n_layers_per_scale, dropout)
            for patch in self.patch_sizes
        ])
        self.cross_scale_bridge = nn.ModuleList([
            CrossScaleBridge(d_model, n_heads, dropout)
            for _ in range(bridge_depth if len(self.patch_sizes) > 1 else 0)
        ])

    def forward(self, x: torch.Tensor):
        if x.ndim != 3 or x.shape[-1] != 1 or x.shape[0] == 0:
            raise ValueError("Expected nonempty single-channel windows with shape (batch, time, 1).")
        if x.shape[1] < max(self.patch_sizes):
            raise ValueError("Window length must be at least the largest patch size.")
        if any((x.shape[1] - patch) // (patch // 2) + 1 > 512
               for patch in self.patch_sizes):
            raise ValueError("A scale has more than 512 tokens; reduce window length.")
        encoded_original = [branch.encode(x) for branch in self.branches]
        encoded = [tokens for tokens, _ in encoded_original]
        for bridge in self.cross_scale_bridge:
            encoded = bridge(encoded)
        scale_errors = torch.stack([
            branch.decode(tokens, original)
            for branch, tokens, (_, original) in zip(self.branches, encoded, encoded_original)
        ], dim=-1)
        return scale_errors, scale_errors.mean(dim=-1)
