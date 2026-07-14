# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Token-based, chunk-form residual TD3 agent with a distilled + frozen RL token.

Follows the reference ``QAgentLatentDistillChunk`` (``use_z_base=False`` variant):

* The obs representation is the **RL token** ``z_rl`` produced by an
  :class:`RLTokenReadout` over the VLA prefix token sequence ``z_1:M``.  The readout
  is **pretrained on offline demos (autoregressive reconstruction) and frozen** for
  the whole of RL — it is NOT trained by the critic.
* At data-collection time ``z_rl`` is precomputed with the frozen readout and stored
  in the replay buffer as ``observation.rl_token``; the raw token sequence is dropped
  (huge memory saving). ``_encode`` prefers the stored ``z_rl`` and only falls back to
  the readout when acting on a live obs.
* Actions are whole chunks flattened to ``action_dim = C * per_step_action_dim``.
  ``observation.base_action`` is the flattened VLA reference chunk ``a~_1:C``;
  ``combined = clamp(a~ + residual, -1, 1)``.  Chunk-level TD target uses ``γ^C``.
* Actor anchors to the VLA reference: ``-Q + β·‖residual‖²``.
"""

from __future__ import annotations

import copy

import torch
from torch import nn

from resfit.rl_finetuning.config.rlpd import QAgentConfig
from resfit.rl_finetuning.off_policy import common_utils
from resfit.rl_finetuning.off_policy.common_utils import utils
from resfit.rl_finetuning.off_policy.networks.rl_token import RLTokenReadout
from resfit.rl_finetuning.off_policy.rl.actor_chunk import ChunkActor
from resfit.rl_finetuning.off_policy.rl.critic_chunk import ChunkCritic


class QAgentTokenChunk(nn.Module):
    VLA_TOKENS_KEY = "observation.vla_tokens"
    RL_TOKEN_KEY = "observation.rl_token"

    def __init__(
        self,
        *,
        token_dim: int,
        max_tokens: int,
        prop_dim: int,
        action_dim: int,
        cfg: QAgentConfig,
        rl_token_dim: int = 512,
        readout_d_model: int = 512,
        readout_layers: int = 3,
        readout_heads: int = 8,
        readout_dropout: float = 0.1,
        distill_lr: float = 1e-4,
        distill_grad_clip_norm: float = 1.0,
        ref_action_dropout: float = 0.5,
    ):
        """
        Args:
            token_dim: width of each VLA prefix token embedding (gemma hidden size).
            max_tokens: max prefix tokens M (upper bound for pos embeds).
            prop_dim: proprio state dim (e.g. 8).
            action_dim: FLATTENED chunk action dim = C * per_step_action_dim.
        """
        super().__init__()
        self.cfg = cfg
        self.token_dim = int(token_dim)
        self.action_dim = int(action_dim)
        self.z_dim = int(rl_token_dim)
        self.residual_actor = True
        self._distill_frozen = False
        self.distill_grad_clip_norm = float(distill_grad_clip_norm)

        self.readout = RLTokenReadout(
            token_dim=token_dim,
            z_dim=rl_token_dim,
            d_model=readout_d_model,
            num_heads=readout_heads,
            num_layers=readout_layers,
            dropout=readout_dropout,
            max_tokens=max_tokens,
        )
        repr_dim = self.readout.repr_dim

        self.critic = ChunkCritic(repr_dim=repr_dim, prop_dim=prop_dim, action_dim=action_dim, cfg=cfg.critic)
        self.actor = ChunkActor(repr_dim, prop_dim, action_dim, cfg.actor, ref_action_dropout=ref_action_dropout)

        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)

        print(common_utils.wrap_ruler("rl-token readout"))
        common_utils.count_parameters(self.readout)
        print(common_utils.wrap_ruler("chunk critic"))
        common_utils.count_parameters(self.critic)
        print(common_utils.wrap_ruler("chunk actor"))
        common_utils.count_parameters(self.actor)

        self.critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=cfg.critic_lr)
        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=cfg.actor_lr)
        # Readout is trained ONLY during the offline distill pretrain, then frozen.
        self.distill_opt = torch.optim.AdamW(self.readout.parameters(), lr=distill_lr)

        self.encoder_scheduler = None
        self.critic_scheduler = None
        self.actor_scheduler = None
        self.lang_encoder = None

        self.critic_target.train(False)
        self.train(True)
        self.to(cfg.device)

    # ------------------------------------------------------------------ #
    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        # Keep the (frozen) readout in eval once frozen.
        self.readout.train(training and not self._distill_frozen)
        assert not self.critic_target.training
        return self

    # ------------------------------------------------------------------ #
    # RL-token access
    # ------------------------------------------------------------------ #
    def _get_tokens(self, obs) -> torch.Tensor:
        t = obs[self.VLA_TOKENS_KEY]
        if t.dim() == 2:
            t = t.unsqueeze(0)
        return t.float()

    @torch.no_grad()
    def encode_rl_token(self, obs) -> torch.Tensor:
        """z_rl = readout.encode(vla_tokens), (B, z_dim). Frozen-readout precompute
        used at collection / offline-populate time to store the compact latent."""
        return self.readout.encode(self._get_tokens(obs))

    def _encode(self, obs) -> torch.Tensor:
        """z_rl (B, z_dim), detached. Prefer the stored ``observation.rl_token``
        (buffer); else compute from the raw ``observation.vla_tokens`` (live obs)."""
        if self.RL_TOKEN_KEY in obs:
            z = obs[self.RL_TOKEN_KEY]
            if z.dim() == 1:
                z = z.unsqueeze(0)
            return z.float().detach()
        return self.readout.encode(self._get_tokens(obs)).detach()

    @staticmethod
    def _ensure_batched(obs):
        obs = copy.copy(obs)
        squeezed = obs["observation.state"].dim() == 1
        if squeezed:
            obs["observation.state"] = obs["observation.state"].unsqueeze(0)
            obs["observation.base_action"] = obs["observation.base_action"].unsqueeze(0)
        return obs, squeezed

    def act(self, obs, *, eval_mode=False, stddev=0.0, cpu=True) -> torch.Tensor:
        assert not self.training
        assert not self.actor.training
        obs, squeezed = self._ensure_batched(obs)
        obs = copy.copy(obs)
        obs["feat"] = self._encode(obs)
        dist = self.actor.forward(obs, stddev)
        action = dist.mean if eval_mode else dist.sample(clip=1)
        if squeezed:
            action = action.squeeze(0)
        action = action.detach()
        return action.cpu() if cpu else action

    # ------------------------------------------------------------------ #
    # Distill pretrain / freeze
    # ------------------------------------------------------------------ #
    def update_distill(self, tokens: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> dict:
        """One readout pretrain step on the AR reconstruction loss (offline only)."""
        self.distill_opt.zero_grad(set_to_none=True)
        loss, m = self.readout.reconstruction_loss(tokens, key_padding_mask)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(self.readout.parameters(), self.distill_grad_clip_norm)
        self.distill_opt.step()
        return {**m, "distill/grad_norm": gn.item()}

    def freeze_distill(self):
        for p in self.readout.parameters():
            p.requires_grad = False
        self.readout.eval()
        self._distill_frozen = True
        print("🧊 RL-token readout frozen (no distill updates during RL)")

    def load_distill_state_dict(self, readout_state: dict, strict: bool = True):
        self.readout.load_state_dict(readout_state, strict=strict)

    # ------------------------------------------------------------------ #
    # RL updates
    # ------------------------------------------------------------------ #
    def _act_target(self, obs, *, eval_mode, stddev, clip):
        assert self.actor_target.training
        dist = self.actor_target.forward(obs, stddev)
        return dist.mean if eval_mode else dist.sample(clip=clip)

    def update_critic(self, obs, action, reward, discount, next_obs, stddev, importance_weights=None):
        with torch.no_grad():
            next_residual = self._act_target(
                next_obs,
                eval_mode=not self.cfg.target_action_noise,
                stddev=stddev,
                clip=self.cfg.stddev_clip,
            )
            next_action = torch.clamp(next_obs["observation.base_action"] + next_residual, -1.0, 1.0)
            target_all = self.critic_target.q_value(next_obs["feat"], next_obs["observation.state"], next_action)
            target_q = (reward + discount * target_all.squeeze(-1)).detach()

        if self.cfg.clip_q_target_to_reward_range:
            target_q = torch.clamp(target_q, min=0, max=1)

        td_errors = None
        loss_type = self.critic.loss_cfg.type
        if loss_type == "hl_gauss":
            _, logits = self.critic(obs["feat"], obs["observation.state"], action, return_logits=True)
            K = logits.shape[0]
            critic_loss = torch.stack([self.critic.hl_loss(logits[i], target_q) for i in range(K)]).mean()
        elif loss_type == "c51":
            _, logits = self.critic(obs["feat"], obs["observation.state"], action, return_logits=True)
            with torch.no_grad():
                _, next_logits = self.critic_target(
                    next_obs["feat"], next_obs["observation.state"], next_action, return_logits=True
                )
                num_heads = min(self.critic.cfg.min_q_heads, next_logits.shape[0])
                idx = torch.randperm(next_logits.shape[0], device=next_logits.device)[:num_heads]
                next_dist = torch.softmax(torch.min(next_logits.index_select(0, idx), dim=0).values, dim=-1)
                dones = (discount == 0.0).float()
                target_dist = self.critic.c51_loss.project_distribution(next_dist, reward, dones, 0.99)
            K = logits.shape[0]
            critic_loss = torch.stack([self.critic.c51_loss(logits[i], target_dist) for i in range(K)]).mean()
        else:
            q_all = self.critic(obs["feat"], obs["observation.state"], action).squeeze(-1)  # [K,B]
            td_errors = torch.abs(q_all - target_q.unsqueeze(0)).mean(dim=0)
            if importance_weights is not None:
                critic_loss = (td_errors**2 * importance_weights).mean()
            else:
                critic_loss = (td_errors**2).mean()

        metrics = {
            "train/critic_qt": target_q.mean().item(),
            "train/critic_loss": critic_loss.item(),
            "_target_q": target_q.detach().cpu(),
        }
        if td_errors is not None:
            metrics["_td_errors"] = td_errors.detach().cpu()

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        metrics["train/critic_grad_norm"] = torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.cfg.critic_grad_clip_norm
        ).item()
        self.critic_opt.step()
        return metrics

    def update_actor(self, obs, stddev):
        metrics = {}
        dist = self.actor.forward(obs, 0.0)  # deterministic policy action for the actor objective
        residual = dist.sample(clip=self.cfg.stddev_clip)

        anchor = self.actor.cfg.action_l2_reg_weight * torch.mean(torch.sum(residual**2, dim=-1))
        combined = torch.clamp(obs["observation.base_action"] + residual, -1.0, 1.0)
        q = self.critic.q_value_for_policy(obs["feat"], obs["observation.state"], combined)
        actor_loss_base = -q.mean()
        actor_loss = actor_loss_base + anchor

        metrics["train/actor_loss_base"] = actor_loss_base.item()
        metrics["train/actor_loss_total"] = actor_loss.item()
        metrics["_actions"] = residual.detach().cpu()
        metrics["_combined_actions"] = combined.detach().cpu()
        if self.actor.cfg.action_l2_reg_weight > 0:
            metrics["train/actor_anchor_penalty"] = anchor.item()
        sigma = getattr(self.actor, "last_sigma", None)
        if sigma is not None:
            metrics["train/scale_sigma_mean"] = sigma.detach().mean().item()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        metrics["train/actor_grad_norm"] = torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.cfg.actor_grad_clip_norm
        ).item()
        self.actor_opt.step()
        return metrics

    def update(self, batch, stddev, update_actor, bc_batch=None, ref_agent=None):
        obs = copy.copy(batch["obs"])
        action = batch["action"]
        reward = batch[("next", "reward")]
        discount = batch["gamma"]
        next_nonterminal = batch["nonterminal"]
        next_obs = copy.copy(batch[("next", "obs")])
        effective_discount = discount * next_nonterminal

        # z_rl comes from the stored observation.rl_token (frozen readout); detached.
        obs["feat"] = self._encode(obs)
        next_obs["feat"] = self._encode(next_obs)

        metrics = {"data/batch_R": reward.mean().item()}
        critic_metric = self.update_critic(
            obs, action, reward, effective_discount, next_obs, stddev,
            importance_weights=batch.get("_weight", None),
        )
        utils.soft_update_params(self.critic, self.critic_target, self.cfg.critic_target_tau)
        metrics.update(critic_metric)

        if not update_actor:
            return metrics

        actor_obs = {
            "feat": obs["feat"],
            "observation.state": obs["observation.state"],
            "observation.base_action": obs["observation.base_action"],
        }
        actor_metric = self.update_actor(actor_obs, stddev)
        utils.soft_update_params(self.actor, self.actor_target, self.cfg.critic_target_tau)
        metrics.update(actor_metric)
        return metrics
