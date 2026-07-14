# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""RL-token readout (faithful to "RL Token: Bootstrapping Online RL with VLA models").

A lightweight transformer encoder-decoder attached to a *frozen* VLA. It reads the
VLA's final-layer token sequence ``z_{1:M}`` together with a learned special token
``e_rl`` and outputs a compact readout ``z_rl`` at the special-token position:

    z_rl = Enc([z_1, ..., z_M, e_rl])_{M+1}

The decoder reconstructs the (stop-gradient) VLA tokens autoregressively from
``z_rl`` — a pure L2 reconstruction bottleneck (NO VAE, NO KL):

    L_ro = E[ sum_i || Dec(z_rl, z̄_{1:i-1})_i - z̄_i ||^2 ],   z̄ = sg(z)

Autoregressive training uses teacher forcing + a causal mask (a single parallel
forward — the decoder is never sampled; it exists only for the loss). After
pretraining on demo data the module is FROZEN; ``encode`` then maps a live VLA token
sequence to ``z_rl`` (the obs representation for the actor-critic).
"""

from __future__ import annotations

import torch
from torch import nn


def _causal_mask(n: int, device) -> torch.Tensor:
    """Upper-triangular (True = disallowed) bool mask for nn attention."""
    return torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), diagonal=1)


class RLTokenReadout(nn.Module):
    """Transformer encoder-decoder readout producing the RL token ``z_rl``."""

    def __init__(
        self,
        token_dim: int,
        z_dim: int = 512,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        max_tokens: int = 4096,
    ):
        super().__init__()
        self.token_dim = int(token_dim)
        self.z_dim = int(z_dim)
        self.d_model = int(d_model)
        self.max_tokens = int(max_tokens)
        # interface parity with the old encoder (repr fed to actor/critic as a vector)
        self.repr_dim = int(z_dim)
        self.patch_repr_dim = int(z_dim)
        self.num_patch = 1

        # ---- Encoder ----
        self.in_proj = nn.Linear(self.token_dim, d_model)
        self.e_rl = nn.Parameter(torch.zeros(1, 1, d_model))
        self.enc_pos = nn.Parameter(torch.zeros(1, self.max_tokens + 1, d_model))
        nn.init.normal_(self.e_rl, std=0.02)
        nn.init.normal_(self.enc_pos, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.enc_ln = nn.LayerNorm(d_model)
        self.to_z = nn.Linear(d_model, self.z_dim)

        # ---- Decoder (causal, teacher-forced reconstruction of z̄_{1:M}) ----
        self.z_to_dec = nn.Linear(self.z_dim, d_model)
        self.tgt_proj = nn.Linear(self.token_dim, d_model)
        self.dec_pos = nn.Parameter(torch.zeros(1, self.max_tokens + 1, d_model))
        nn.init.normal_(self.dec_pos, std=0.02)
        dec_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.dec_ln = nn.LayerNorm(d_model)
        self.out_head = nn.Linear(d_model, self.token_dim)

    def encode(self, tokens: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """``tokens``: (B, M, token_dim) [or (M, token_dim)]. Returns z_rl: (B, z_dim)."""
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)
        tokens = tokens.float()
        B, M, _ = tokens.shape
        assert M <= self.max_tokens, f"seq len {M} > max_tokens {self.max_tokens}"
        x = self.in_proj(tokens)
        e = self.e_rl.expand(B, -1, -1)
        x = torch.cat([x, e], dim=1) + self.enc_pos[:, : M + 1]
        pad = None
        if key_padding_mask is not None:
            pad = torch.cat(
                [key_padding_mask, torch.zeros(B, 1, dtype=torch.bool, device=tokens.device)], dim=1
            )
        h = self.encoder(x, src_key_padding_mask=pad)
        return self.to_z(self.enc_ln(h[:, -1]))

    def reconstruction_loss(
        self, tokens: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict]:
        """L2 AR reconstruction of sg(tokens) from z_rl. Returns (loss, metrics)."""
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)
        tokens = tokens.float()
        B, M, _ = tokens.shape
        target = tokens.detach()
        z_rl = self.encode(tokens, key_padding_mask)

        z0 = self.z_to_dec(z_rl).unsqueeze(1)
        shifted = self.tgt_proj(target[:, : M - 1])
        dec_in = torch.cat([z0, shifted], dim=1) + self.dec_pos[:, :M]
        attn_mask = _causal_mask(M, tokens.device)
        h = self.decoder(dec_in, mask=attn_mask, src_key_padding_mask=key_padding_mask)
        pred = self.out_head(self.dec_ln(h))

        sq = ((pred - target) ** 2).sum(dim=-1)  # (B, M)
        if key_padding_mask is not None:
            valid = (~key_padding_mask).float()
            recon = (sq * valid).sum() / valid.sum().clamp_min(1.0)
        else:
            recon = sq.mean()
        metrics = {
            "rl_token/recon_loss": recon.detach().item(),
            "rl_token/z_abs_mean": z_rl.detach().abs().mean().item(),
        }
        return recon, metrics
