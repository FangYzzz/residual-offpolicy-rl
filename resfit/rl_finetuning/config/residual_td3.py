# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore

from resfit.rl_finetuning.config.rlpd import ActorConfig, QAgentConfig, RLPDAlgoConfig, RLPDDexmgConfig


@dataclass
class OfflineDataConfig:
    name: str = "ankile/robomimic-mh-can-image"
    num_episodes: int | None = 300
    horizon: int = 400
    # Offline data action labeling options
    use_base_policy_for_base_actions: bool = True
    # Normalization safeguards
    min_action_range: float = 1e-1  # Minimum range for any action dimension to prevent normalization blow-up
    min_state_std: float = 1e-1  # Minimum std for any state dimension to prevent normalization blow-up


@dataclass
class WandBConfig:
    project: str = "robomimic-can-residual-td3"
    mode: str = "online"
    entity: str | None = None
    notes: str | None = None
    continue_run_id: str | None = None
    resume_checkpoint_run: bool = True  # True 不重开 wandb, False: 重开 wandb
    name: str | None = None
    group: str | None = None


@dataclass
class BasePolicyConfig:
    wandb_id: str = "dexmg-bc/o2h7mdwe"
    wt_type: str = "best"
    wt_version: str = "latest"


@dataclass
class RLTokenConfig:
    """RL-Token bottleneck and cache settings (disabled for legacy trainers)."""

    enabled: bool = False
    # Optional smoke-test limit. When set, this overrides
    # offline_data.num_episodes for embedding collection, VAE training, and
    # offline replay construction so every stage uses the same subset.
    offline_num_episodes: int | None = 460
    obs_key: str = "observation.rl_token"
    vla_embedding_key: str = "vla_embedding"
    embedding_cache_dir: str = "vla_embedding_cache"
    checkpoint_dir: str = "rl_token_checkpoints"
    force_recollect_embeddings: bool = False
    # Number of embeddings buffered in RAM before an atomic disk shard is
    # written. Raw VLA sequences are never accumulated for the full dataset.
    embedding_shard_size: int = 32
    embedding_storage_dtype: str = "int8"
    force_retrain: bool = False
    token_dim: int = 512
    model_dim: int = 1024
    encoder_layers: int = 2
    decoder_layers: int = 2
    num_heads: int = 8
    dropout: float = 0.1
    batch_size: int = 4
    # Upper bound for padded sequence tokens per VAE minibatch. This
    # automatically reduces batch_size for long pi0 prefix sequences.
    max_tokens_per_batch: int = 1024
    epochs: int = 50
    validation_fraction: float = 0.1
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-4
    split_seed: int = 0
    max_train_samples_per_epoch: int | None = 10_000
    max_validation_samples: int | None = 2_000
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    beta_kl: float = 1e-4
    num_workers: int = 0
    # Request final VLA prefix features (all camera views + language tokens).
    request_field: str = "return_vla_embedding"


@dataclass
class ResidualTD3AlgoConfig(RLPDAlgoConfig):
    # ------------------------------------------------------------------
    # Critic warmup phase ----------------------------------------------
    # ------------------------------------------------------------------
    # Number of critic-only updates before training the actor
    critic_warmup_steps: int = 10_000

    # ------------------------------------------------------------------
    # Random action exploration -----------------------------------------
    # ------------------------------------------------------------------
    # Scale for random action noise during initial exploration phase
    # Actions are sampled as: rand_actions = torch.rand(...) * 2 * random_action_noise_scale - random_action_noise_scale
    random_action_noise_scale: float = 0.2  # Default: uniform in [-1, 1]

    # Whether to use base policy + noise (True) or pure uniform noise (False) during warmup
    # Note: Environment wrapper always applies base_action + residual_action
    # True: residual_action = noise (resulting in base_action + noise)
    # False: residual_action = pure_random - base_action (resulting in pure_random)
    use_base_policy_for_warmup: bool = True

    # Whether a resumed training process should discard the per-task Beta
    # success-rate statistics and restart every task from 0.5. Set to False
    # to restore those statistics from the training checkpoint.
    reset_task_success_rates_on_restart: bool = False  # Restore per-task success rates from the checkpoint on resume.

    # ------------------------------------------------------------------
    # Standard deviation schedule -------------------------------------------
    # ------------------------------------------------------------------
    stddev_max: float = 0.006 # 0.05
    stddev_min: float = 0.001 # 0.05
    stddev_step: int = 100_000 # 300_000

    # Progressive clipping schedule for the residual actions
    # I.e., starts clipping linearly from 0 to action scale over progressive_clipping_steps steps
    progressive_clipping_steps: int = 0


# -----------------------------------------------------------------------------
# Top-level experiment config --------------------------------------------------
# -----------------------------------------------------------------------------
@dataclass
class ResidualTD3DexmgConfig(RLPDDexmgConfig):
    actor_name: str | None = None  # Inferred from base policy config

    # ------------------------------------------------------------------
    # Algorithm & optimisation
    # ------------------------------------------------------------------
    algo: ResidualTD3AlgoConfig = field(default_factory=ResidualTD3AlgoConfig)

    # ------------------------------------------------------------------
    # Network architectures
    # ------------------------------------------------------------------
    agent: QAgentConfig = field(
        default_factory=lambda: QAgentConfig(
            actor_lr=1e-6,
            critic_lr=1e-4,
            critic_target_tau=0.005,
            actor=ActorConfig(
                action_scale=0.1,
                actor_last_layer_init_scale=0.0,  # imp for residual
            ),
        )
    )

    # ------------------------------------------------------------------
    # Offline dataset
    # ------------------------------------------------------------------
    offline_data: OfflineDataConfig | None = field(default_factory=OfflineDataConfig)

    # ------------------------------------------------------------------
    # Base policy
    # ------------------------------------------------------------------
    base_policy: BasePolicyConfig = field(default_factory=BasePolicyConfig)

    rl_token: RLTokenConfig = field(default_factory=RLTokenConfig)

    # ------------------------------------------------------------------
    # Weights & Biases logging
    # ------------------------------------------------------------------
    wandb: WandBConfig = field(default_factory=WandBConfig)

    # ------------------------------------------------------------------
    # Logging / checkpointing
    # ------------------------------------------------------------------
    # Save robot debug-camera frames and GPT before/after scene images for
    # evaluation, online training, and online replay-buffer warmup.
    save_images: bool = True

    eval_interval_every_steps: int = 40_000  ### 10_000

    # Whether to run an evaluation pass before training begins (at step 0)
    eval_first: bool = True  ### True

    resume: bool = False
    resume_checkpoint: str | None = None
    # Set this only when the periodic evaluation at resume_checkpoint's
    # global_step completed before the previous process stopped. The collector
    # continues at the next chunk instead of repeating that step/evaluation;
    # later evaluation intervals are unchanged.
    skip_completed_eval_on_resume: bool = False
    checkpoint_interval: int = 1000  ### 5000
    # Replay buffers must be checkpointed together with the model so that
    # global_step and buffer/online_size stay consistent after resuming.
    save_replay_on_checkpoint: bool = True
    # Keep checkpoints and artifacts in the original run directory when
    # resuming, instead of creating run_<new timestamp>_<name> each time.
    reuse_run_dir_on_resume: bool = True
    # Checkpoints are required for future resume, so preserve the run directory
    # after a normal training completion unless cleanup is explicitly requested.
    cleanup_run_dir_on_finish: bool = False
    save_online_rb_interval: int = 1000  ### 5000
    send_transitions_len : int = 1  ### 
    chunk_len: int = 5  ################## # residual action chunk length H (actor/critic act_dim = H * per_step_dim)

@dataclass
class ResidualTD3CanConfig(ResidualTD3DexmgConfig):
    task: str = "Can"

    offline_data: OfflineDataConfig = field(
        default_factory=lambda: OfflineDataConfig(
            name="ankile/robomimic-mh-can-image",
            num_episodes=300,
        )
    )

    base_policy: BasePolicyConfig = field(
        default_factory=lambda: BasePolicyConfig(
            wandb_id="robomimic-can-bc/sdo8cku7",
        )
    )

    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="robomimic-can-residual-td3"))


@dataclass
class ResidualTD3SquareConfig(ResidualTD3DexmgConfig):
    task: str = "Square"

    offline_data: OfflineDataConfig = field(
        default_factory=lambda: OfflineDataConfig(
            name="ankile/robomimic-mh-square-image",
            num_episodes=300,
        )
    )

    base_policy: BasePolicyConfig = field(
        default_factory=lambda: BasePolicyConfig(
            wandb_id="robomimic-square-bc/dzbkdpwp",
        )
    )

    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="robomimic-square-residual-td3"))


@dataclass
class ResidualTD3BoxCleanConfig(ResidualTD3DexmgConfig):
    task: str = "TwoArmBoxCleanup"

    rl_camera: list[str] = field(
        default_factory=lambda: [
            "observation.images.agentview",
            "observation.images.robot0_eye_in_hand",
            "observation.images.robot1_eye_in_hand",
        ]
    )

    algo: ResidualTD3AlgoConfig = field(
        default_factory=lambda: ResidualTD3AlgoConfig(
            total_timesteps=500_000,
        )
    )

    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="dexmg-box-clean-residual-td3"))

    offline_data: OfflineDataConfig = field(
        default_factory=lambda: OfflineDataConfig(
            name="ankile/dexmg-two-arm-box-cleanup",
            num_episodes=1_000,
        )
    )
    base_policy: BasePolicyConfig = field(
        default_factory=lambda: BasePolicyConfig(
            wandb_id="TODO",
            wt_type="best",
            wt_version="latest",
        )
    )

@dataclass
class ResidualTD3FrankaComplexConfig(ResidualTD3DexmgConfig):
    # task: str = "FrankaTomatoPnP"
    # task: str = "pick up the tomato and place it into the bowl"

    rl_camera: list[str] = field(
        default_factory=lambda: [
            # "observation.images.exterior_image_1_left",
            "observation.images.exterior_image_2_left",
            "observation.images.wrist_image_left",
        ]
    )

    algo: ResidualTD3AlgoConfig = field(
        default_factory=lambda: ResidualTD3AlgoConfig(
            total_timesteps=500_000,
        )
    )

    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="franka-complex-residual-td3"))

    offline_data: OfflineDataConfig = field(
        default_factory=lambda: OfflineDataConfig(
            name="/home/yuan/self_vla/tele_op/lerobot/dataset_7050_7050_5050_7050",
            num_episodes=460,
            horizon=400,   # 这里改成真实 episode 长度
        )
    )
    base_policy: BasePolicyConfig = field(
        default_factory=lambda: BasePolicyConfig(
            wandb_id="TODO",
            wt_type="best",
            wt_version="latest",
        )
    )

@dataclass
class ResidualTD3FrankaCubeConfig(ResidualTD3DexmgConfig):
    task: str = "pick up the cube and place it into the bowl"

    rl_camera: list[str] = field(
        default_factory=lambda: [
            "observation.images.exterior_image_1_left",
            "observation.images.exterior_image_2_left",
            "observation.images.wrist_image_left",
        ]
    )

    algo: ResidualTD3AlgoConfig = field(
        default_factory=lambda: ResidualTD3AlgoConfig(
            total_timesteps=500_000,
        )
    )

    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="franka-cube-residual-td3"))

    offline_data: OfflineDataConfig = field(
        default_factory=lambda: OfflineDataConfig(
            name="/home/yuan/self_vla/tele_op/lerobot/cube_fix_in50_out30",
            # name="/home/yuan/self_vla/tele_op/lerobot/cube_in50_out50",
            # name="/home/yuan/self_vla/tele_op/lerobot/cube_done",
            num_episodes=1_000,
            horizon=140,   # 这里改成真实 episode 长度
        )
    )
    base_policy: BasePolicyConfig = field(
        default_factory=lambda: BasePolicyConfig(
            wandb_id="TODO",
            wt_type="best",
            wt_version="latest",
        )
    )


@dataclass
class ResidualTD3CoffeeConfig(ResidualTD3BoxCleanConfig):
    task: str = "TwoArmCoffee"

    rl_camera: list[str] = field(
        default_factory=lambda: [
            "observation.images.agentview",
            "observation.images.robot0_eye_in_left_hand",
            "observation.images.robot0_eye_in_right_hand",
        ]
    )

    algo: ResidualTD3AlgoConfig = field(
        default_factory=lambda: ResidualTD3AlgoConfig(
            total_timesteps=500_000,
        )
    )

    wandb: WandBConfig = field(
        default_factory=lambda: WandBConfig(project="dexmg-coffee-residual-td3", notes="all cameras")
    )

    offline_data: OfflineDataConfig = field(
        default_factory=lambda: OfflineDataConfig(
            name="ankile/dexmg-two-arm-coffee",
            num_episodes=1_000,
        )
    )
    base_policy: BasePolicyConfig = field(
        default_factory=lambda: BasePolicyConfig(
            wandb_id="dexmg-bc/o2h7mdwe",
            wt_type="best",
            wt_version="latest",
        )
    )

@dataclass
class ResidualTD3TwoArmCanSortConfig(ResidualTD3BoxCleanConfig):
    task: str = "TwoArmCanSortRandom"

    rl_camera: list[str] = field(
        default_factory=lambda: [
            "observation.images.frontview",
            "observation.images.robot0_eye_in_left_hand",
            "observation.images.robot0_eye_in_right_hand",
        ]
    )

    wandb: WandBConfig = field(default_factory=lambda: WandBConfig(project="dexmg-cansort-residual-td3"))

    offline_data: OfflineDataConfig = field(
        default_factory=lambda: OfflineDataConfig(
            name="ankile/dexmg-two-arm-can-sort-random",
            num_episodes=1_000,
        )
    )
    base_policy: BasePolicyConfig = field(
        default_factory=lambda: BasePolicyConfig(
            wandb_id="TODO",
            wt_type="best",
            wt_version="latest",
        )
    )


# -----------------------------------------------------------------------------
# Register with Hydra
# -----------------------------------------------------------------------------
cs = ConfigStore.instance()
cs.store(name="residual_td3_dexmg_config", node=ResidualTD3DexmgConfig)
cs.store(name="residual_td3_can_config", node=ResidualTD3CanConfig)
cs.store(name="residual_td3_square_config", node=ResidualTD3SquareConfig)
cs.store(name="residual_td3_box_clean_config", node=ResidualTD3BoxCleanConfig)
cs.store(name="residual_td3_coffee_config", node=ResidualTD3CoffeeConfig)
cs.store(name="residual_td3_two_arm_cansort_config", node=ResidualTD3TwoArmCanSortConfig)
cs.store(name="residual_td3_franka_complex_config", node=ResidualTD3FrankaComplexConfig)
cs.store(name="residual_td3_franka_cube_config", node=ResidualTD3FrankaCubeConfig)
