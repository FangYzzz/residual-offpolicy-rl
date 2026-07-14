# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Chunk-level residual actor conditioned on the RL token.

Token-based, chunk-form counterpart of ``off_policy.rl.actor.Actor`` for the
RL-Token method.  Differences from the per-step ViT actor:

* Perception is a single RL-token vector ``z_rl`` (obs["feat"]), not image patches,
  so the trunk is a plain MLP (no SpatialEmb).
* The action is a whole chunk, flattened to ``action_dim = C * per_step_action_dim``.
  ``observation.base_action`` is the flattened VLA reference chunk ``a~_1:C``.
* Reference-action dropout: during training the reference chunk fed to the trunk is
  masked (per sample) with probability ``ref_action_dropout`` so the policy does not
  collapse onto the VLA reference (paper, sec. method).

The learnable per-dim residual scale head is kept identical in spirit to
``actor.Actor`` (``sigma(s) in (0, s_max]``, ``scaled_mu = sigma * tanh(mu)``).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from resfit.rl_finetuning.config.rlpd import ActorConfig
from resfit.rl_finetuning.off_policy.common_utils import utils
from resfit.rl_finetuning.off_policy.rl.actor import build_fc


class ChunkActor(nn.Module):
    def __init__(
        self,
        repr_dim: int,
        prop_dim: int,
        action_dim: int,
        cfg: ActorConfig,
        *,
        ref_action_dropout: float = 0.5,
    ):
        super().__init__()
        self.cfg = cfg
        self.action_dim = action_dim  # = C * per_step_action_dim
        self.prop_dim = prop_dim
        self.ref_action_dropout = float(ref_action_dropout)

        # Compress the RL token into a feature (kept for parity with Actor).
        layers = [nn.Linear(repr_dim, cfg.feature_dim)]
        if cfg.use_layer_norm:
            layers.append(nn.LayerNorm(cfg.feature_dim))
        layers.extend([nn.Dropout(cfg.dropout), nn.ReLU()])
        self.compress = nn.Sequential(*layers)

        # Trunk sees [feat, state, reference_chunk].
        policy_in_dim = cfg.feature_dim + prop_dim + action_dim

        # ---- Learnable residual scale head (shares the policy network) ----
        self.scale_head_enabled: bool = bool(getattr(cfg, "scale_head_enabled", True))
        if self.scale_head_enabled:
            self.s_max = float(getattr(cfg, "scale_head_max", cfg.action_scale))
            init_value = float(getattr(cfg, "scale_head_init_value", cfg.action_scale))
            assert 0.0 < init_value < self.s_max, (
                f"scale_head_init_value ({init_value}) must satisfy 0 < init < s_max ({self.s_max})"
            )
            self._scale_per_dim = bool(getattr(cfg, "scale_head_per_dim", True))
            self._sigma_dim = action_dim if self._scale_per_dim else 1
            p0 = init_value / self.s_max
            self._sigma_bias_init = math.log(p0 / (1.0 - p0))
        else:
            self.s_max = None
            self._scale_per_dim = False
            self._sigma_dim = 0
            self._sigma_bias_init = 0.0

        policy_out_dim = action_dim + self._sigma_dim
        self.policy = build_fc(
            policy_in_dim,
            cfg.hidden_dim,
            policy_out_dim,
            num_layer=cfg.num_layers,
            layer_norm=1,
            dropout=cfg.dropout,
            use_layer_norm=cfg.use_layer_norm,
            final_activation=None,
        )

        self.last_sigma: torch.Tensor | None = None
        self._initialize_weights(cfg)

    def _initialize_weights(self, cfg: ActorConfig):
        intermediate_init = cfg.actor_intermediate_layer_init_distribution
        if cfg.orth and intermediate_init == "default":
            intermediate_init = "orthogonal"

        if cfg.orth:
            self.compress.apply(utils.orth_weight_init)
        else:
            utils.apply_initialization_to_network(self.compress, intermediate_init)

        utils.apply_initialization_to_network(self.policy, intermediate_init, exclude_final_layer=True)

        final_layer = self._find_final_linear(self.policy)
        if cfg.actor_last_layer_init_scale is not None and final_layer is not None:
            utils.initialize_layer_weights(
                final_layer,
                cfg.actor_last_layer_init_distribution,
                cfg.actor_last_layer_init_scale,
            )
        if self.scale_head_enabled and final_layer is not None:
            with torch.no_grad():
                final_layer.weight[self.action_dim :, :].zero_()
                final_layer.bias[self.action_dim :].fill_(self._sigma_bias_init)

    @staticmethod
    def _find_final_linear(module: nn.Module) -> nn.Linear | None:
        for m in reversed(list(module.modules())):
            if isinstance(m, nn.Linear):
                return m
        return None

    def forward(self, obs: dict[str, torch.Tensor], std: float):
        feat = self.compress(obs["feat"])  # [B, feature_dim]

        base_chunk = obs["observation.base_action"]  # [B, action_dim]
        if self.training and self.ref_action_dropout > 0.0:
            keep = (torch.rand(base_chunk.size(0), 1, device=base_chunk.device) >= self.ref_action_dropout).float()
            ref_input = base_chunk * keep
        else:
            ref_input = base_chunk

        all_input = [feat]
        if self.prop_dim > 0:
            all_input.append(obs["observation.state"])
        all_input.append(ref_input)
        policy_input = torch.cat(all_input, dim=-1)

        out: torch.Tensor = self.policy(policy_input)
        mu = torch.tanh(out[..., : self.action_dim])

        if self.scale_head_enabled:
            sigma = self.s_max * torch.sigmoid(out[..., self.action_dim :])
            if not self._scale_per_dim:
                sigma = sigma.expand(-1, self.action_dim)
            scaled_mu = sigma * mu
            self.last_sigma = sigma
        else:
            scaled_mu = mu * self.cfg.action_scale
            self.last_sigma = None

        return utils.TruncatedNormal(scaled_mu, std)
