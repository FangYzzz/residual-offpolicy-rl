# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Config for token-based, chunk-form residual TD3 (RL-Token method).

Extends :class:`ResidualTD3DexmgConfig` with the RL-token / chunk knobs and flips
a few defaults that only make sense for the token pipeline:

* ``offline_fraction = 0`` — the offline LeRobot dataset has no stored VLA tokens,
  so token-mode training is online-only.
* ``n_step = 1`` — the chunk itself is the temporal extension; chunk discounting
  (``γ^C``) is applied via the buffer transform's ``gamma`` in the training script.
* language disabled — the prompt is already inside the VLA prefix tokens ``z_1:M``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore

from resfit.rl_finetuning.config.rlpd import ActorConfig, CriticConfig, LanguageConfig, QAgentConfig
from resfit.rl_finetuning.config.residual_td3 import (
    BasePolicyConfig,
    OfflineDataConfig,
    ResidualTD3AlgoConfig,
    ResidualTD3DexmgConfig,
    WandBConfig,
)


@dataclass
class RLTokenConfig:
    # Chunk length C (VLA emits 50 steps; the per-step base policy executes 35).
    chunk_len: int = 35
    per_step_action_dim: int = 8

    # VLA prefix token embeddings z_1:M.
    # token_dim (gemma hidden size) and max_tokens (M) are auto-detected from the
    # first server response at startup; the values here are only fallbacks.
    token_dim: int = 2048
    max_tokens: int = 1024

    # RL-token readout (transformer encoder-decoder) sizes.
    rl_token_dim: int = 512
    readout_d_model: int = 512
    readout_layers: int = 3
    readout_heads: int = 8
    readout_dropout: float = 0.1

    # ---- Distill pretrain of the readout on offline demos (then frozen) ----
    distill_pretrain_steps: int = 3000
    distill_lr: float = 1e-4
    distill_grad_clip_norm: float = 1.0
    distill_pretrain_batch_size: int = 16
    distill_pretrain_max_frames: int = 2000
    freeze_after_pretrain: bool = True
    # Optional explicit ckpt path; if None, an auto config-hashed cache path is used.
    distill_ckpt: str | None = None

    # Offline chunk-transition stride (in env steps). Default = chunk_len (non-overlap).
    offline_stride: int = 35

    # Reference-chunk dropout in the actor.
    ref_action_dropout: float = 0.5


@dataclass
class ResidualTD3TokenChunkConfig(ResidualTD3DexmgConfig):
    task: str = "put the cube into the bowl"

    token: RLTokenConfig = field(default_factory=RLTokenConfig)

    rl_camera: list[str] = field(
        default_factory=lambda: [
            "observation.images.exterior_image_1_left",
            "observation.images.exterior_image_2_left",
            "observation.images.wrist_image_left",
        ]
    )

    # Mixed offline+online (RLPD-style). Chunk-level discounting (γ^C) is applied via
    # the buffer transform in the training script; n_step=1 (the chunk is the temporal
    # extension). learning_starts / buffer_size count CHUNK transitions, not env steps.
    algo: ResidualTD3AlgoConfig = field(
        default_factory=lambda: ResidualTD3AlgoConfig(
            total_timesteps=500_000,
            offline_fraction=0.5,
            n_step=1,
            critic_warmup_steps=0,
            learning_starts=200,
            buffer_size=50_000,
        )
    )

    # Reuse actor/critic hyperparameters; disable the language head.
    agent: QAgentConfig = field(
        default_factory=lambda: QAgentConfig(
            actor_lr=1e-4,
            critic_lr=1e-4,
            critic_target_tau=0.005,
            language=LanguageConfig(enabled=False),
            actor=ActorConfig(
                action_scale=0.1,
                actor_last_layer_init_scale=0.0,
                action_l2_reg_weight=1.0,  # anchor beta to the VLA reference chunk
            ),
            critic=CriticConfig(),
        )
    )

    offline_data: OfflineDataConfig = field(
        default_factory=lambda: OfflineDataConfig(
            name="/home/yuan/self_vla/tele_op/lerobot/cube_fix_in50_out30",
            num_episodes=1_000,
            horizon=140,
        )
    )

    base_policy: BasePolicyConfig = field(default_factory=BasePolicyConfig)

    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="franka-cube-residual-td3-token"))

    # One chunk transition sent to the learner at a time.
    send_transitions_len: int = 1


cs = ConfigStore.instance()
cs.store(name="residual_td3_token_chunk_config", node=ResidualTD3TokenChunkConfig)
