# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Chunk-level critic conditioned on the RL token.

Token-based counterpart of ``off_policy.rl.critic.Critic``.  The observation
feature is a single RL-token vector ``z_rl`` (obs["feat"], shape [B, repr_dim]),
so the spatial-embedding patch trunk is replaced by a plain MLP trunk.  The action
is the whole flattened chunk (``action_dim = C * per_step_action_dim``).

The public API (``forward(..., return_logits=)``, ``q_value``,
``q_value_for_policy``, ``loss_cfg``, ``hl_loss``, ``c51_loss``) matches
``critic.Critic`` so ``QAgentTokenChunk`` can drive it with the same update code as
``QAgentLang``.  The ensemble heads reuse ``HeadMLP`` and the loss objects reuse
``HLGaussLoss`` / ``C51Loss`` from ``critic.py``.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.func import functional_call, stack_module_state, vmap

from resfit.rl_finetuning.config.rlpd import CriticConfig
from resfit.rl_finetuning.off_policy.common_utils import utils
from resfit.rl_finetuning.off_policy.rl.critic import C51Loss, HeadMLP, HLGaussLoss


class ChunkQEnsemble(nn.Module):
    """Shared MLP trunk + K vmap'd MLP heads over [z_rl, prop, action]."""

    def __init__(
        self,
        *,
        repr_dim: int,
        prop_dim: int,
        action_dim: int,
        emb_dim: int,
        hidden_dim: int,
        orth: int,
        output_dim: int = 1,
        num_heads: int = 2,
        num_layers: int = 2,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.prop_dim = prop_dim
        self.action_dim = action_dim

        in_dim = repr_dim + prop_dim + action_dim
        input_layers = [nn.Linear(in_dim, emb_dim)]
        if use_layer_norm:
            input_layers.append(nn.LayerNorm(emb_dim))
        input_layers.append(nn.ReLU(inplace=True))
        self.input_proj = nn.Sequential(*input_layers)

        self.num_heads = num_heads
        head_in = emb_dim
        heads = [HeadMLP(head_in, hidden_dim, output_dim, num_layers, use_layer_norm) for _ in range(num_heads)]
        self.params, self.buffers = stack_module_state(heads)
        self._head_template = HeadMLP(head_in, hidden_dim, output_dim, num_layers, use_layer_norm)

        for name, param in self.params.items():
            self.register_parameter(f"_vmap_param_{name.replace('.', '_')}", nn.Parameter(param))
        for name, buffer in self.buffers.items():
            self.register_buffer(f"_vmap_buffer_{name.replace('.', '_')}", buffer)

        self._init_per_head_params(orth)

    def _init_per_head_params(self, orth: bool):
        with torch.no_grad():
            if orth:
                for key, param in self.params.items():
                    if "weight" in key and param.dim() >= 2:
                        for h in range(self.num_heads):
                            if param[h].dim() >= 2:
                                utils.orth_weight_init(param[h])

    def forward(self, feat: torch.Tensor, prop: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        parts = [feat]
        if self.prop_dim > 0:
            parts.append(prop)
        parts.append(action)
        z = self.input_proj(torch.cat(parts, dim=-1))  # [B, emb_dim]

        current_params = {n: getattr(self, f"_vmap_param_{n.replace('.', '_')}") for n in self.params}
        current_buffers = {n: getattr(self, f"_vmap_buffer_{n.replace('.', '_')}") for n in self.buffers}

        def f_one_head(p, b, z_input):
            return functional_call(self._head_template, (p, b), (z_input,))

        return vmap(f_one_head, in_dims=(0, 0, None))(current_params, current_buffers, z)


class ChunkCritic(nn.Module):
    def __init__(self, repr_dim, prop_dim, action_dim, cfg: CriticConfig):
        super().__init__()
        self.cfg = cfg
        self.loss_cfg = cfg.loss

        output_dim = self.loss_cfg.n_bins if self.loss_cfg.type in {"hl_gauss", "c51"} else 1
        num_q = getattr(cfg, "num_q", 2)

        self.q_ensemble = ChunkQEnsemble(
            repr_dim=repr_dim,
            prop_dim=prop_dim,
            action_dim=action_dim,
            emb_dim=cfg.spatial_emb,
            hidden_dim=cfg.hidden_dim,
            orth=cfg.orth,
            output_dim=output_dim,
            num_heads=num_q,
            num_layers=cfg.num_layers,
            use_layer_norm=cfg.use_layer_norm,
        )

        if self.loss_cfg.type == "hl_gauss":
            sigma_val = None if self.loss_cfg.sigma < 0 else self.loss_cfg.sigma
            self.hl_loss = HLGaussLoss(
                min_value=self.loss_cfg.v_min,
                max_value=self.loss_cfg.v_max,
                num_bins=self.loss_cfg.n_bins,
                sigma=sigma_val,
            )
        elif self.loss_cfg.type == "c51":
            self.c51_loss = C51Loss(
                v_min=self.loss_cfg.v_min,
                v_max=self.loss_cfg.v_max,
                num_atoms=self.loss_cfg.n_bins,
            )

    @staticmethod
    def _logits_to_q(probs, support):
        return (probs * support).sum(-1, keepdim=True)

    def forward(self, feat, prop, act, *, return_logits: bool = False):
        logits_per_head = self.q_ensemble(feat, prop, act)  # [num_q, B, out_dim]

        if self.loss_cfg.type == "hl_gauss":
            q_per_head = self._logits_to_q(torch.softmax(logits_per_head, dim=-1), self.hl_loss.bin_centers)
            return (q_per_head, logits_per_head) if return_logits else q_per_head
        if self.loss_cfg.type == "c51":
            q_per_head = self.c51_loss.logits_to_q_value(logits_per_head)
            return (q_per_head, logits_per_head) if return_logits else q_per_head
        return logits_per_head  # MSE: already scalars per head [num_q, B, 1]

    def q_value(self, feat, prop, act):
        q_out = self.forward(feat, prop, act)
        num_heads = min(self.cfg.min_q_heads, q_out.shape[0])
        idx = torch.randperm(q_out.shape[0], device=q_out.device)[:num_heads]
        return torch.min(q_out.index_select(0, idx), dim=0).values

    def q_value_for_policy(self, feat, prop, act):
        q_out = self.forward(feat, prop, act)
        if self.cfg.policy_gradient_type == "ensemble_mean":
            return q_out.mean(dim=0)
        if self.cfg.policy_gradient_type == "min_random_pair":
            num_heads = min(self.cfg.min_q_heads, q_out.shape[0])
            idx = torch.randperm(q_out.shape[0], device=q_out.device)[:num_heads]
            return torch.min(q_out.index_select(0, idx), dim=0).values
        if self.cfg.policy_gradient_type == "q1":
            return q_out[0]
        raise ValueError(f"Unknown policy_gradient_type: {self.cfg.policy_gradient_type}")
