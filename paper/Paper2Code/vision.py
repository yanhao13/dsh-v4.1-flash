"""DeepSeek-ViT vision encoder + pixel-unshuffle + MLP projector.

The multimodal pathway (Section 2.1.1) runs each image through a from-scratch
ViT, then applies a 3x3 pixel-unshuffle to cut the visual token count by 9x,
then maps features to the language backbone's hidden dimension via a 2-layer
MLP projector. DeepSeek-ViT uses 2D-RoPE, RMSNorm, SwiGLU, and a linear (rather
than convolutional) patch embedding for Muon compatibility.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import RMSNorm, precompute_2d_rope_freqs, apply_rotary_emb


class DeepSeekViT(nn.Module):
    def __init__(self, n_layers: int = 32, hidden: int = 1024, n_heads: int = 16,
                 patch: int = 14, n_px: int = 3):
        super().__init__()
        self.patch = patch
        self.hidden = hidden
        self.head_dim = hidden // n_heads
        self.n_heads = n_heads
        self.patch_embed = nn.Linear(3 * patch * patch, hidden, bias=False)
        self.blocks = nn.ModuleList([
            _ViTBlock(hidden, n_heads) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(hidden)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 3, H, W) -> (B, L, hidden) visual features."""
        b, c, h, w = images.shape
        ph, pw = h // self.patch, w // self.patch
        # Extract patches (simple unfold).
        patches = images.unfold(2, self.patch, self.patch).unfold(3, self.patch, self.patch)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(b, ph * pw, -1)
        x = self.patch_embed(patches)  # (B, L, hidden)
        freqs = precompute_2d_rope_freqs(self.head_dim, ph, pw).reshape(ph * pw, self.head_dim)
        freqs = freqs.to(x.device)
        for blk in self.blocks:
            x = blk(x, freqs)
        return self.norm(x)


class _ViTBlock(nn.Module):
    def __init__(self, hidden: int, n_heads: int):
        super().__init__()
        self.head_dim = hidden // n_heads
        self.n_heads = n_heads
        self.norm1 = RMSNorm(hidden)
        self.norm2 = RMSNorm(hidden)
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.o = nn.Linear(hidden, hidden, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden * 4, bias=False),
            nn.SiLU(),
            nn.Linear(hidden * 4, hidden, bias=False),
        )

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        b, l, h = x.shape
        qkv = self.qkv(self.norm1(x)).view(b, l, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = apply_rotary_emb(q, freqs.unsqueeze(0).unsqueeze(2))
        k = apply_rotary_emb(k, freqs.unsqueeze(0).unsqueeze(2))
        attn = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                              v.transpose(1, 2)).transpose(1, 2)
        x = x + self.o(attn.reshape(b, l, h))
        x = x + self.mlp(self.norm2(x))
        return x


class PixelUnshuffle(nn.Module):
    """3x3 pixel-unshuffle: reduces spatial grid by 9x, channels by 9x."""

    def __init__(self, scale: int = 3):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
        """x: (B, L, C); grid: (ph, pw) -> (B, L//9, C*9)."""
        b, l, c = x.shape
        ph, pw = grid
        x = x.reshape(b, ph, pw, c).permute(0, 3, 1, 2)  # (B, C, ph, pw)
        x = F.pixel_unshuffle(x, self.scale)             # (B, C*9, ph/3, pw/3)
        x = x.flatten(2).transpose(1, 2)                 # (B, L/9, C*9)
        return x


class MLPProjector(nn.Module):
    """2-layer MLP projector mapping visual features to the LLM hidden dim."""

    def __init__(self, in_dim: int, hidden: int = 5120, n_layers: int = 2):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden, bias=False), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden, bias=False), nn.SiLU()]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
