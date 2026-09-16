"""Top-level DeepSeek-V4.1-Flash model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DeepSeekV41Config
from .norm import RMSNorm
from .ced import CED
from .vision import DeepSeekViT, PixelUnshuffle, MLPProjector


class DeepSeekV41Flash(nn.Module):
    def __init__(self, cfg: DeepSeekV41Config):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.hidden)
        self.vision = DeepSeekViT(cfg.vit_n_layers, cfg.vit_hidden, cfg.vit_n_heads,
                                  cfg.vit_patch, cfg.vit_pixel_unshuffle)
        self.unshuffle = PixelUnshuffle(cfg.vit_pixel_unshuffle)
        # Projector input: 9x the ViT hidden dim after 3x3 pixel-unshuffle.
        self.projector = MLPProjector(cfg.vit_hidden * (cfg.vit_pixel_unshuffle ** 2),
                                      cfg.hidden, cfg.projector_n_layers)
        self.backbone = CED(cfg)
        self.norm = RMSNorm(cfg.hidden)
        self.lm_head = nn.Linear(cfg.hidden, cfg.vocab_size, bias=False)
        # Tie weights.
        self.lm_head.weight = self.token_embedding.weight

    def embed_images(self, images: list[torch.Tensor]) -> list[torch.Tensor]:
        """Encode a list of images into visual embedding sequences."""
        out = []
        for img in images:
            feat = self.vision(img)                      # (1, L, vit_hidden)
            _, l, c = feat.shape
            ph = pw = int(round(l ** 0.5))
            feat = self.unshuffle(feat, (ph, pw))        # (1, L/9, c*9)
            feat = self.projector(feat)                  # (1, L/9, hidden)
            out.append(feat.squeeze(0))
        return out

    def forward(self, token_ids: torch.Tensor,
                images: list[torch.Tensor] | None = None,
                image_positions: list[int] | None = None) -> torch.Tensor:
        """token_ids: (B, T). Returns (B, T, vocab) logits."""
        h = self.token_embedding(token_ids)
        backbone_ids = token_ids
        if images is not None:
            # Insert visual embeddings at the given positions (single image per
            # position for the reference implementation).
            assert image_positions is not None
            emb = self.embed_images(images)
            parts, id_parts = [], []
            cursor = 0
            for img_emb, pos in zip(emb, image_positions):
                parts.append(h[:, cursor:pos])
                id_parts.append(token_ids[:, cursor:pos])
                parts.append(img_emb.unsqueeze(0).expand(h.shape[0], -1, -1))
                # Sentinels (0) for the visual tokens, so Engram hashing aligns.
                id_parts.append(torch.zeros(h.shape[0], img_emb.shape[0],
                                            dtype=token_ids.dtype, device=token_ids.device))
                cursor = pos + 1  # skip the placeholder image token
            parts.append(h[:, cursor:])
            id_parts.append(token_ids[:, cursor:])
            h = torch.cat([p for p in parts if p.shape[1] > 0], dim=1)
            backbone_ids = torch.cat([p for p in id_parts if p.shape[1] > 0], dim=1)
        h = self.backbone(h, backbone_ids)
        return self.lm_head(self.norm(h))
