"""DeepSeekMoE with modality-aware auxiliary-loss-free load balancing.

Every feed-forward layer uses standard DeepSeekMoE: one shared expert plus
``n_routed_experts`` routed experts, each a SwiGLU MLP with a clamp threshold
of 10 (OpenAI, 2025). Routing selects the top-``n_activated_experts`` experts
per token. Load balancing follows DeepSeek-V3's auxiliary-loss-free scheme
(Wang et al., 2024a) extended with separate expert-wise bias sets for text and
image tokens.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import RMSNorm


class SwiGLUExpert(nn.Module):
    """A single SwiGLU expert with clamping."""

    def __init__(self, hidden: int, inter: int, clamp: float = 10.0):
        super().__init__()
        self.gate = nn.Linear(hidden, inter, bias=False)
        self.up = nn.Linear(hidden, inter, bias=False)
        self.down = nn.Linear(inter, hidden, bias=False)
        self.clamp = clamp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.gate(x)
        g = g.clamp(max=self.clamp)
        act = F.silu(g) * self.up(x)
        return self.down(act)


class DeepSeekMoE(nn.Module):
    def __init__(self, hidden: int, n_routed: int, n_shared: int, n_act: int,
                 inter: int, clamp: float = 10.0):
        super().__init__()
        self.n_act = n_act
        self.shared = SwiGLUExpert(hidden, inter * n_shared if n_shared else inter, clamp) \
            if n_shared else None
        self.experts = nn.ModuleList([SwiGLUExpert(hidden, inter, clamp) for _ in range(n_routed)])
        self.router = nn.Linear(hidden, n_routed, bias=False)
        # Auxiliary-loss-free biases: one set per modality (text=0, image=1).
        self.register_buffer("route_bias_text", torch.zeros(n_routed))
        self.register_buffer("route_bias_image", torch.zeros(n_routed))
        self.register_buffer("_text_cnt", torch.zeros(n_routed))
        self.register_buffer("_img_cnt", torch.zeros(n_routed))

    def forward(self, x: torch.Tensor, modality: torch.Tensor | None = None) -> torch.Tensor:
        """Route and mix experts.

        Args:
            x: (B, T, hidden).
            modality: (B, T) int tensor, 0=text, 1=image. Defaults to all-text.
        """
        b, t, h = x.shape
        if modality is None:
            modality = torch.zeros(b, t, dtype=torch.long, device=x.device)

        logits = self.router(x)  # (B, T, E)
        # Select the modality-specific correction bias per token.
        bias_text = self.route_bias_text[None, None, :]
        bias_image = self.route_bias_image[None, None, :]
        mask = (modality == 1).unsqueeze(-1).float()
        bias = bias_text * (1 - mask) + bias_image * mask
        logits = logits + bias

        topk_logits, topk_idx = logits.topk(self.n_act, dim=-1)  # (B, T, K)
        weights = F.softmax(topk_logits, dim=-1).to(x.dtype)

        out = torch.zeros_like(x)
        # Gather expert outputs (reference clarity over raw speed).
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)                       # (B, T, K)
            if not mask.any():
                continue
            token_sel = mask.any(dim=-1)                 # (B, T)
            xtok = x[token_sel]                          # (N_i, hidden)
            yi = expert(xtok)                            # (N_i, hidden)
            w = weights.masked_fill(~mask, 0).sum(dim=-1)[token_sel]  # (N_i,)
            out[token_sel] += yi * w.unsqueeze(-1)

        if self.shared is not None:
            out = out + self.shared(x)
        return out

    def update_bias(self, speed: float = 0.001):
        """Auxiliary-loss-free bias update: nudge over/under-loaded experts."""
        # Uses counters accumulated over the last step (set externally).
        avg = (self._text_cnt.mean() + self._img_cnt.mean()) / 2
        for name, cnt in (("text", self._text_cnt), ("image", self._img_cnt)):
            if name == "text":
                target = self.route_bias_text
            else:
                target = self.route_bias_image
            # Move bias toward balance: overloaded experts get less bias.
            target.add_(-speed * (cnt - avg))
        self._text_cnt.zero_()
        self._img_cnt.zero_()
