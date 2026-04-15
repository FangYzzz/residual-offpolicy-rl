# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

import os

# Cap all BLAS/OpenMP threadpools (critical to set before importing numpy/torch)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
# Stop threads from spin-waiting
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("KMP_AFFINITY", "granularity=fine,compact,1,0")

import hashlib
import json
import logging
import pprint
import random
import shutil
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import torch.nn.functional as F

import hydra
import numpy as np
import tensordict
import torch
import torchrl
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from omegaconf import OmegaConf
from tensordict import TensorDict
from torch.utils.data import DataLoader
from torchrl.data import LazyTensorStorage, ReplayBuffer, TensorDictPrioritizedReplayBuffer
from tqdm import tqdm

import wandb
from resfit.dexmg.environments.dexmg import create_vectorized_env
from resfit.lerobot.policies.act.configuration_act import ACTConfig
from resfit.lerobot.policies.act.modeling_act import ACTPolicy
from resfit.lerobot.utils.load_policy import download_policy_from_wandb, load_policy
from resfit.rl_finetuning.config.residual_td3 import ResidualTD3DexmgConfig
from resfit.rl_finetuning.off_policy.common_utils import utils
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.utils.dtype import to_uint8
from resfit.rl_finetuning.utils.evaluate_dexmg import run_dexmg_evaluation
from resfit.rl_finetuning.utils.evaluate_franka import run_franka_evaluation
from resfit.rl_finetuning.utils.hugging_face import (
    _hf_download_buffer,
    _hf_upload_buffer,
    optimized_replay_buffer_dumps,
    optimized_replay_buffer_loads,
)
from resfit.rl_finetuning.utils.normalization import ActionScaler, StateStandardizer
from resfit.rl_finetuning.utils.rb_transforms import MultiStepTransform
from resfit.rl_finetuning.wrappers.residual_env_wrapper import BasePolicyVecEnvWrapper
from residual_client import ResidualClient

# -----------------------------------------------------------------------------
# Timing utility --------------------------------------------------------------
# -----------------------------------------------------------------------------
class TrainingTimer:
    """Simple timing utility for measuring training stage proportions."""

    def __init__(self):
        self.times = defaultdict(list)
        self.reset_time = time.perf_counter()

    @contextmanager
    def time(self, stage_name: str):
        """Context manager to time a specific training stage."""
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        self.times[stage_name].append(elapsed)

    def get_timing_stats(self) -> dict[str, float]:
        """Get timing statistics as percentages of total time."""
        if not self.times:
            return {}

        # Calculate total time across all stages
        total_time = sum(sum(times) for times in self.times.values())
        if total_time == 0:
            return {}

        # Calculate percentages and averages
        stats = {}
        for stage_name, times_list in self.times.items():
            stage_total = sum(times_list)
            stage_avg = stage_total / len(times_list) if times_list else 0
            stage_percentage = (stage_total / total_time) * 100

            stats[f"timing/{stage_name}_percentage"] = stage_percentage
            stats[f"timing/{stage_name}_avg_ms"] = stage_avg * 1000  # Convert to ms
            stats[f"timing/{stage_name}_total_s"] = stage_total

        return stats

    def reset(self):
        """Reset all timing data."""
        self.times = defaultdict(list)
        self.reset_time = time.perf_counter()


# -----------------------------------------------------------------------------
# Logging configuration -------------------------------------------------------
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Hugging Face buffer cache helpers (global) ---------------------------------
# -----------------------------------------------------------------------------
OFFLINE_HF_REPO = os.environ.get("HF_OFFLINE_BUFFER_REPO", None)
ONLINE_HF_REPO = os.environ.get("HF_ONLINE_BUFFER_REPO", None)

if OFFLINE_HF_REPO is not None:
    logger.info(f"Using offline buffer from {OFFLINE_HF_REPO}")
if ONLINE_HF_REPO is not None:
    logger.info(f"Using online buffer from {ONLINE_HF_REPO}")

# Generic environment variable (shared across algorithms) -------------------
# ``CACHE_DIR`` specifies the root folder for **all** local caches.
# Falls back to the current directory if unset.
_CACHE_ROOT = Path(os.environ.get("CACHE_DIR", ".")).expanduser().resolve()

# Dedicated sub-folders for the different cache types -----------------------
OFFLINE_CACHE_DIR = _CACHE_ROOT / "offline_buffer_cache"
ONLINE_CACHE_DIR = _CACHE_ROOT / "online_buffer_cache"


# -----------------------------------------------------------------------------
# Repository-local imports ------------------------------------------------------
# -----------------------------------------------------------------------------
# os.environ["MUJOCO_GL"] = "egl"

# if "MUJOCO_EGL_DEVICE_ID" in os.environ:
#     del os.environ["MUJOCO_EGL_DEVICE_ID"]

def process_image_batch(obs_dict, image_keys, enc_type= "vit", rb=False):
    """
    将 obs_dict 里的多个图像键统一处理成 [3, out_size, out_size]
    输入每张图预期是 [H, W, C] 或 [1, H, W, C]
    输出每张图是 float32, [3, out_size, out_size], range [0, 1]
    """
    imgs = []
    # original_shapes = {}
    
    for k in image_keys:
        x = obs_dict[k]
        # original_shapes[k] = x.shape

        # 支持 [1,H,W,C] 或 [1,C,H,W]
        if x.ndim == 4 and x.shape[0] == 1:
            x = x.squeeze(0)
        if x.ndim != 3:
            raise ValueError(f"{k} expected 3 dims [H,W,C], got shape={x.shape}")

        # HWC[224,224,3] -> CHW[3,224,224]
        if x.shape[-1] == 3:
            x = x.permute(2, 0, 1)
        elif x.shape[0] == 3:
            pass
        else:
            raise ValueError(f"{k} is neither HWC nor CHW, got shape={x.shape}")

        # uint8 -> float
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        else:
            x = x.float()
        imgs.append(x)

    # [N,3,H,W]
    imgs = torch.stack(imgs, dim=0)
    
    # 只有尺寸不匹配时才 resize
    out_size = 224 if enc_type == "siglip" else 84  # cfg.agent.enc_type == "vit"
    if imgs.shape[-2] != out_size or imgs.shape[-1] != out_size:
        imgs = F.interpolate(
            imgs,
            size=(out_size, out_size),
            mode="bilinear",
            align_corners=False,
        )
    # print("process_image_batch-->imgs.shape:", imgs.shape)  # [3,3,224,224] [3, 3, 84, 84]

    # 写回 obs_dict
    if rb:
        for i, k in enumerate(image_keys):
            obs_dict[k] = imgs[i].contiguous()
    else:  # train
        for i, k in enumerate(image_keys):
            obs_dict[k] = imgs[i].contiguous().unsqueeze(0)

    return obs_dict

def save_training_checkpoint(
    ckpt_path: Path,
    agent,
    online_rb,
    offline_rb,
    global_step: int,
    best_eval_success_rate: float,
    training_cum_time: float,
    episode_count: int,
    actor_updates: int,
    run_name: str,
    cfg,
    save_replay_buffers: bool = False,
):
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "global_step": global_step,
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "actor_opt": agent.actor_opt.state_dict(),
        "critic_opt": agent.critic_opt.state_dict(),
        "best_eval_success_rate": best_eval_success_rate,
        "training_cum_time": training_cum_time,
        "episode_count": episode_count,
        "actor_updates": actor_updates,
        "run_name": run_name,
        "cfg": OmegaConf.to_container(cfg, resolve=True),

        # RNG states
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }

    torch.save(checkpoint, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    if save_replay_buffers:
        rb_dir = ckpt_path.parent / f"{ckpt_path.stem}_replay"
        rb_dir.mkdir(parents=True, exist_ok=True)

        online_dir = rb_dir / "online_rb"
        offline_dir = rb_dir / "offline_rb"

        if online_dir.exists():
            shutil.rmtree(online_dir)
        if offline_dir.exists():
            shutil.rmtree(offline_dir)

        optimized_replay_buffer_dumps(online_rb, online_dir)
        optimized_replay_buffer_dumps(offline_rb, offline_dir)

        meta = {
            "online_size": len(online_rb),
            "offline_size": len(offline_rb),
        }
        with open(rb_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"Saved replay buffers to {rb_dir}")

def load_training_checkpoint(
    ckpt_path: str | Path,
    agent,
    online_rb,
    offline_rb,
    device: torch.device,
    load_replay_buffers: bool = False,
):
    ckpt_path = Path(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    agent.actor.load_state_dict(checkpoint["actor"])
    agent.critic.load_state_dict(checkpoint["critic"])
    agent.actor_opt.load_state_dict(checkpoint["actor_opt"])
    agent.critic_opt.load_state_dict(checkpoint["critic_opt"])

    global_step = checkpoint.get("global_step", 0)
    best_eval_success_rate = checkpoint.get("best_eval_success_rate", 0.0)
    training_cum_time = checkpoint.get("training_cum_time", 0.0)
    episode_count = checkpoint.get("episode_count", 0)
    actor_updates = checkpoint.get("actor_updates", 0)
    run_name = checkpoint.get("run_name", None)

    # Restore RNG states
    if "python_random_state" in checkpoint:
        random.setstate(checkpoint["python_random_state"])
    if "numpy_random_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_random_state"])
    if "torch_random_state" in checkpoint and checkpoint["torch_random_state"] is not None:
        torch_rng_state = checkpoint["torch_random_state"]
        if not isinstance(torch_rng_state, torch.Tensor):
            torch_rng_state = torch.tensor(torch_rng_state, dtype=torch.uint8)
        else:
            torch_rng_state = torch_rng_state.to(dtype=torch.uint8, device="cpu")
        torch.set_rng_state(torch_rng_state)
    if torch.cuda.is_available() and checkpoint.get("cuda_random_state") is not None:
        cuda_rng_state = checkpoint["cuda_random_state"]
        fixed_cuda_states = []
        for s in cuda_rng_state:
            if not isinstance(s, torch.Tensor):
                s = torch.tensor(s, dtype=torch.uint8)
            else:
                s = s.to(dtype=torch.uint8, device="cpu")
            fixed_cuda_states.append(s)
        torch.cuda.set_rng_state_all(fixed_cuda_states)

    if load_replay_buffers:
        rb_dir = ckpt_path.parent / f"{ckpt_path.stem}_replay"
        online_dir = rb_dir / "online_rb"
        offline_dir = rb_dir / "offline_rb"

        if online_dir.exists():
            online_rb.sampler._empty()
            optimized_replay_buffer_loads(online_rb, online_dir)
            print(f"Loaded online replay buffer from {online_dir}, size={len(online_rb)}")
        else:
            print(f"Online replay buffer dir not found: {online_dir}")

        if offline_dir.exists():
            offline_rb.sampler._empty()
            optimized_replay_buffer_loads(offline_rb, offline_dir)
            print(f"Loaded offline replay buffer from {offline_dir}, size={len(offline_rb)}")
        else:
            print(f"Offline replay buffer dir not found: {offline_dir}")

    print(f"Loaded checkpoint from {ckpt_path}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    return {
        "global_step": global_step,
        "best_eval_success_rate": best_eval_success_rate,
        "training_cum_time": training_cum_time,
        "episode_count": episode_count,
        "actor_updates": actor_updates,
        "run_name": run_name,
    }



def _add_transitions_to_buffer(
    *,
    obs: dict,
    next_obs: dict,
    actions: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
    info: dict,
    device: torch.device,
    image_keys: list[str],
    lowdim_keys: list[str],
    num_envs: int,
    online_rb: TensorDictPrioritizedReplayBuffer,
    enc_type: str,
) -> None:
    """Helper function to create transitions and add them to the replay buffer.

    Handles terminal observations correctly and convert images to uint8 for storage.
    """
    obs_keys_set = set(image_keys) | set(lowdim_keys)
    for i in range(num_envs):
        # Handle terminal observation (same logic as main loop)
        if done[i] and "final_obs" in info and info["final_obs"][i] is not None:
            final_obs_dict = info["final_obs"][i]
            next_obs_i = {k: torch.as_tensor(v, device=device) for k, v in final_obs_dict.items()}
        else:
            next_obs_i = {k: v[i] for k, v in next_obs.items()}

        curr_obs_i = {k: v[i] for k, v in obs.items()}

        # Keep only relevant keys & convert images to uint8 for storage
        curr_obs_i = {k: v for k, v in curr_obs_i.items() if k in obs_keys_set}
        next_obs_i = {k: v for k, v in next_obs_i.items() if k in obs_keys_set}
        curr_obs_i = process_image_batch(curr_obs_i, image_keys, enc_type, rb=True)  ###
        next_obs_i = process_image_batch(next_obs_i, image_keys, enc_type, rb=True)
        to_uint8(curr_obs_i, image_keys)
        to_uint8(next_obs_i, image_keys)

        td = TensorDict(
            {
                "obs": TensorDict(curr_obs_i, batch_size=[]),
                "next": TensorDict(
                    {
                        "obs": TensorDict(next_obs_i, batch_size=[]),
                        "done": done[i],
                        "reward": reward[i],
                    },
                    batch_size=[],
                ),
                "action": actions[i],
                "_priority": torch.tensor(10.0, dtype=torch.float32),  # High initial priority for new samples
            },
            batch_size=[],
        ).unsqueeze(0)

        online_rb.add(td)
    
    # obs_keys_set = set(image_keys) | set(lowdim_keys)
    # for i in range(num_envs):
    #     # Handle terminal observation (same logic as main loop)
    #     if done and "final_obs" in info and info["final_obs"][i] is not None:
    #         final_obs_dict = info["final_obs"][i]
    #         next_obs_i = {k: torch.as_tensor(v, device=device) for k, v in final_obs_dict.items()}
    #     else:
    #         next_obs_i = {k: v for k, v in next_obs.items()}

    #     curr_obs_i = {k: v for k, v in obs.items()}

    #     # Keep only relevant keys & convert images to uint8 for storage
    #     curr_obs_i = {k: v.squeeze() for k, v in curr_obs_i.items() if k in obs_keys_set}
    #     next_obs_i = {k: v.squeeze() for k, v in next_obs_i.items() if k in obs_keys_set}
    #     curr_obs_i = process_image_batch(curr_obs_i, image_keys)
    #     next_obs_i = process_image_batch(next_obs_i, image_keys)
    #     to_uint8(curr_obs_i, image_keys)
    #     to_uint8(next_obs_i, image_keys)

    #     td = TensorDict(
    #         {
    #             "obs": TensorDict(curr_obs_i, batch_size=[]),
    #             "next": TensorDict(
    #                 {
    #                     "obs": TensorDict(next_obs_i, batch_size=[]),
    #                     "done": done.squeeze(),
    #                     "reward": reward.squeeze(),
    #                 },
    #                 batch_size=[],
    #             ),
    #             "action": actions.squeeze(),
    #             "_priority": torch.tensor(10.0, dtype=torch.float32),  # High initial priority for new samples
    #         },
    #         batch_size=[],
    #     ).unsqueeze(0)

    #     online_rb.add(td)

    


# -----------------------------------------------------------------------------
# Main training loop -----------------------------------------------------------
# -----------------------------------------------------------------------------
def main(cfg: ResidualTD3DexmgConfig):
    print("cfg.algo.random_action_noise_scale:::::", cfg.algo.random_action_noise_scale)
    print("cfg.task:::::", cfg.task)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    enc_type=cfg.agent.enc_type
    print("agent.enc_type:::::", enc_type)

    # Enable performance optimizations
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ---------------------------------------------------------------------
    # Load the behaviour-cloning policy that will serve as the "base" policy
    # for residual learning.
    # ---------------------------------------------------------------------
    assert "base_policy" in cfg, "Base policy configuration is required"
    # policy_dir, _ = download_policy_from_wandb(
    #     cfg.base_policy.wandb_id,
    #     step=cfg.base_policy.wt_type,
    #     artifact_version=cfg.base_policy.wt_version,
    # )

    # base_policy: ACTPolicy = load_policy(policy_dir)   # 改成 server client
    # base_policy.to(device)
    # base_policy.eval()
    # eval_base_policy: ACTPolicy = load_policy(policy_dir)
    # eval_base_policy.to(device)
    # eval_base_policy.eval()

    # Extract the configuration from base policy
    # base_cfg = base_policy.config

    # if isinstance(base_cfg, ACTConfig):
    #     cfg.actor_name = "residual_act"
    # else:
    #     raise ValueError(f"Unknown base policy type: {type(base_cfg)}")

    # Load dataset and get normalization functions early
    print("Loading dataset and setting up normalization...")
    dataset = LeRobotDataset(cfg.offline_data.name)  # todo
    print("stats keys:", list(dataset.meta.stats.keys()))
    # dataset_path = "/home/yuan/self_vla/tele_op/lerobot/pnp_pi05_tomato_3cams_wrist"
    # dataset = LeRobotDataset(dataset_path)

    # just for check
    # print("stats keys:", list(dataset.meta.stats.keys()))
    # stats keys: ['exterior_image_1_left', 'eef_position', 'timestamp', 'gripper_position', 'exterior_image_2_left', 
    # 'joint_actions', 'eef_actions', 'score', 'frame_index', 'index', 'task_index', 'episode_index', 'joint_position', 'wrist_image_left']
    # sample = dataset[0]
    # print("sample keys:", list(sample.keys()))
    # sample keys: ['exterior_image_1_left', 'exterior_image_2_left', 'wrist_image_left', 'joint_position', 'eef_position', 'gripper_position', 
    # 'joint_actions', 'eef_actions', 'score', 'timestamp', 'frame_index', 'episode_index', 'index', 'task_index', 'task']


    # Create action scaler from dataset statistics
    action_scaler = ActionScaler.from_dataset_stats(
        # action_stats=dataset.meta.stats["action"],
        action_stats=dataset.meta.stats["eef_actions"],
        action_scale=cfg.agent.actor.action_scale,
        min_range_per_dim=cfg.offline_data.min_action_range,
        device=device,
    )

    # Create state standardizer from dataset statistics
    # state_standardizer = StateStandardizer.from_dataset_stats(
    #     state_stats=dataset.meta.stats["observation.state"],
    #     min_std=cfg.offline_data.min_state_std,
    #     device=device,
    # )
    def concat_state_stats(stats_a: dict, stats_b: dict) -> dict:
        out = {}
        for k in ["mean", "std", "min", "max"]:
            if k in stats_a and k in stats_b:
                a = torch.as_tensor(stats_a[k], dtype=torch.float32)
                b = torch.as_tensor(stats_b[k], dtype=torch.float32)
                out[k] = torch.cat([a, b], dim=-1)
        return out
    
    state_standardizer = StateStandardizer.from_dataset_stats(
        state_stats=concat_state_stats(
            dataset.meta.stats["eef_position"],
            dataset.meta.stats["gripper_position"],
        ),
        min_std=cfg.offline_data.min_state_std,
        device=device,
    )
    base_policy = ResidualClient(main_host="127.0.0.1", main_port=8009, action_scaler= action_scaler, state_standardizer=state_standardizer) # todo
    eval_base_policy = ResidualClient(main_host="127.0.0.1", main_port=8009, action_scaler= action_scaler,state_standardizer= state_standardizer) # todo
    # def get_envs(
    #     env_name: str,
    #     num_envs: int,
    #     base_policy: ACTPolicy,
    #     device: str,
    #     video_key: str,
    #     debug: bool,
    #     action_scaler: ActionScaler,
    #     state_standardizer: StateStandardizer,
    # ):
    #     assert action_scaler is not None, "action_scaler must be provided for consistent normalization"
    #     assert state_standardizer is not None, "state_standardizer must be provided for consistent normalization"

    #     # Create the vectorized environment
    #     vec_env = create_vectorized_env(
    #         env_name=env_name,
    #         num_envs=num_envs,
    #         device=device,
    #         video_key=video_key,
    #         debug=debug,
    #     )

    #     # Wrap it with the base policy wrapper
    #     return BasePolicyVecEnvWrapper(
    #         vec_env=vec_env,
    #         base_policy=base_policy,
    #         action_scaler=action_scaler,
    #         state_standardizer=state_standardizer,
    #     )

    # ---------------------------------------------------------------------
    # Seeding (must be done before environment creation) ------------------
    # ---------------------------------------------------------------------
    if cfg.seed is None:
        cfg.seed = random.randint(0, 2**32 - 1)

    # Comprehensive seeding for reproducibility
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # CUDA seeding for multi-GPU reproducibility
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # Set deterministic behavior
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    print(f"Set random seed to {cfg.seed}")

    # ---------------------------------------------------------------------
    # Environment setup ----------------------------------------------------
    # ---------------------------------------------------------------------
    assert cfg.num_envs == 1, "Only support 1 environment for now because of how n_step is implemented"
    # env = get_envs(
    #     env_name=cfg.task,
    #     num_envs=cfg.num_envs,
    #     base_policy=base_policy,
    #     device=device_str,
    #     video_key=cfg.video_key,
    #     debug=cfg.debug,
    #     action_scaler=action_scaler,
    #     state_standardizer=state_standardizer,
    # )
    cfg.eval_num_envs = min(cfg.eval_num_envs, cfg.eval_num_episodes)
    num_cpus_available = os.cpu_count() - 1 if os.cpu_count() is not None else 1
    cfg.eval_num_envs = min(num_cpus_available, cfg.eval_num_envs)

    # eval_env = get_envs(
    #     env_name=cfg.task,
    #     num_envs=cfg.eval_num_envs,
    #     base_policy=eval_base_policy,
    #     device=device_str,
    #     video_key=cfg.video_key,
    #     debug=cfg.debug,
    #     action_scaler=action_scaler,
    #     state_standardizer=state_standardizer,
    # )

    # Seed environments explicitly for reproducibility
    # if hasattr(env, "seed"):
    #     env.seed(cfg.seed)
    # if hasattr(eval_env, "seed"):
    #     eval_env.seed(cfg.seed + 1)  # Use different seed for eval env to avoid correlation

    # ---------------------------------------------------------------------
    # Observation / action dimensions -------------------------------------
    # ---------------------------------------------------------------------
    # Determine which image keys (camera observations) will be used. The
    # configuration can specify either a single camera name (str) or a list of
    # names.
    if isinstance(cfg.rl_camera, str): # todo
        image_keys: list[str] = [cfg.rl_camera]
    else:
        image_keys = list(cfg.rl_camera)
    assert isinstance(image_keys, list)
    # lowdim_dim = env.observation_space["observation.state"].shape[1]  # todo define shape 
    # img_c, img_h, img_w = env.observation_space[image_keys[0]].shape[1:]  # todo define dim shape
    # action_dim = env.action_space.shape[1]  # todo define dim 
    # lowdim_keys = ["observation.state", "observation.base_action"]  # todo define key

    lowdim_dim = 8  # 8+8
    # img_c, img_h, img_w =3, 224, 224
    img_c, img_h, img_w =3, 84, 84
    action_dim = 8
    lowdim_keys = ["observation.state", "observation.base_action"]

    # ---------------------------------------------------------------------
    # Networks ------------------------------------------------------------
    # ---------------------------------------------------------------------
    agent = QAgent(
        obs_shape=(img_c, img_h, img_w),
        prop_shape=(lowdim_dim,),
        action_dim=action_dim,
        rl_cameras=image_keys,
        cfg=cfg.agent,
        residual_actor=True,  # Enable residual actor mode
    ) 

    # horizon = env.vec_env.metadata["horizon"]  # todo
    horizon = cfg.offline_data.horizon

    # Set up actor learning rate warmup
    actor_updates = 0
    if cfg.algo.actor_lr_warmup_steps > 0:
        print(
            f"Actor LR warmup enabled: 0.0 -> {cfg.agent.actor_lr:.2e} "
            f"over {cfg.algo.actor_lr_warmup_steps} actor updates"
        )

    # ---------------------------------------------------------------------
    # Replay buffers -------------------------------------------------------
    # ---------------------------------------------------------------------
    # -----------------------------------------------------------------
    # Use TensorDictPrioritizedReplayBuffer for unified PER support
    # For uniform sampling, we'll use alpha=0 and beta=0, and never update priorities
    # -----------------------------------------------------------------
    alpha = cfg.algo.priority_alpha if cfg.algo.sampling_strategy == "prioritized_replay" else 0.0
    beta = cfg.algo.priority_beta if cfg.algo.sampling_strategy == "prioritized_replay" else 0.0

    online_batch_size = int(cfg.algo.batch_size * (1 - cfg.algo.offline_fraction))
    offline_batch_size = int(cfg.algo.batch_size * cfg.algo.offline_fraction)

    if cfg.algo.offline_fraction == 0.0:
        print("Online-only training mode: offline_fraction=0.0")

    # Use TensorDictPrioritizedReplayBuffer with optimized prefetching
    online_rb = TensorDictPrioritizedReplayBuffer(
        storage=LazyTensorStorage(max_size=cfg.algo.buffer_size, device="cpu"),
        alpha=alpha,
        beta=beta,
        eps=1e-6,  # Small epsilon added to priorities to prevent zero values
        priority_key="_priority",
        transform=MultiStepTransform(n_steps=cfg.algo.n_step, gamma=cfg.algo.gamma),
        pin_memory=True,
        prefetch=cfg.algo.prefetch_batches,  # Add prefetching
        batch_size=online_batch_size,
    )

    # ------------------------------------------------------------------
    # Caching layer for online replay buffer ----------------------------
    # ------------------------------------------------------------------
    online_cache_meta = {
        "task": cfg.task,  # todo!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        "image_keys": image_keys,
        "n_step": cfg.algo.n_step,
        "gamma": cfg.algo.gamma,
        "horizon": horizon,
        "size": cfg.algo.learning_starts,
        "sampling_strategy": cfg.algo.sampling_strategy,
        "buffer_size": cfg.algo.buffer_size,
        "batch_size": online_batch_size,
        # Include random action noise scale to prevent mixing data from different noise levels
        "random_action_noise_scale": cfg.algo.random_action_noise_scale,
        # Normalization parameters for consistency
        "min_action_range": cfg.offline_data.min_action_range,
        "min_state_std": cfg.offline_data.min_state_std,
        "normalized_actions": True,
        # Library versions for compatibility
        "torchrl_version": torchrl.__version__,
        "tensordict_version": tensordict.__version__,
    }
    if cfg.algo.sampling_strategy == "prioritized_replay":
        online_cache_meta["priority_alpha"] = cfg.algo.priority_alpha
        online_cache_meta["priority_beta"] = cfg.algo.priority_beta

    pprint.pprint(online_cache_meta)
    _online_meta_str = json.dumps(online_cache_meta, sort_keys=True)
    online_cache_hash = hashlib.sha1(_online_meta_str.encode()).hexdigest()[:8]  # noqa: S324
    # Base local path for the online buffer ------------------------------
    online_cache_dir = ONLINE_CACHE_DIR / online_cache_hash

    # Attempt to download/extract from HF every run (no-op if already cached)
    dl_dir = None
    if ONLINE_HF_REPO is not None:
        print(f"Attempting to download online buffer {online_cache_hash} from {ONLINE_HF_REPO}...")
        dl_dir = _hf_download_buffer(ONLINE_HF_REPO, online_cache_hash, ONLINE_CACHE_DIR)
    if dl_dir is not None:
        online_cache_dir = dl_dir

    loaded_online_from_cache = False
    if online_cache_dir.exists():
        print(f"{online_cache_dir} found on disk. Attempting to load...")
        online_rb.sampler._empty()
        optimized_replay_buffer_loads(online_rb, online_cache_dir)
        loaded_online_from_cache = True
        print(f"Loaded online buffer from cache at {online_cache_dir} (size={len(online_rb)})")

    # Offline data is required for normalization, but can be unused for training if offline_fraction=0
    assert cfg.offline_data is not None and cfg.offline_data.num_episodes is not None

    # Dataset and normalization already loaded above - use existing dataset

    # Use actual dataset metadata for precise buffer sizing
    if cfg.offline_data.num_episodes is not None:
        # Only use subset of episodes if specified
        total_frames = sum(
            dataset.meta.episodes[ep_idx]["length"]
            for ep_idx in range(min(cfg.offline_data.num_episodes, dataset.meta.total_episodes))
        )
        num_episodes = cfg.offline_data.num_episodes
    else:
        # Use entire dataset
        total_frames = dataset.meta.total_frames
        num_episodes = dataset.meta.total_episodes

    # Calculate transitions: each episode contributes (episode_length - 1) transitions
    estimated_transitions = max(0, total_frames - num_episodes)

    print("Dataset buffer sizing:")
    print(f"  Total frames to process: {total_frames}")
    print(f"  Number of episodes: {num_episodes}")
    print(f"  Estimated transitions: {estimated_transitions}")

    # Calculate buffer size for simplified approach (1 transition per frame pair)
    max_offline_transitions = (
        estimated_transitions if cfg.algo.offline_fraction > 0.0 else 1
    )  # Minimum size for online-only mode
    if cfg.algo.offline_fraction > 0.0:
        print(f"Offline buffer sized for GT-as-base approach: {max_offline_transitions} transitions")
    else:
        print("Online-only mode: creating minimal offline buffer (unused)")

    offline_rb = TensorDictPrioritizedReplayBuffer(
        storage=LazyTensorStorage(max_size=max_offline_transitions, device="cpu"),
        alpha=alpha,
        beta=beta,
        eps=1e-6,  # Small epsilon added to priorities to prevent zero values
        priority_key="_priority",
        transform=MultiStepTransform(n_steps=cfg.algo.n_step, gamma=cfg.algo.gamma),
        pin_memory=True,
        prefetch=cfg.algo.prefetch_batches,  # Add prefetching
        batch_size=max(offline_batch_size, 1),  # Ensure batch_size is at least 1
    )
    
    # Normalization functions already defined above - use them

    # ------------------------------------------------------------------
    # Convert offline dataset episodes into transitions and fill buffer
    # ------------------------------------------------------------------
    def _populate_offline_buffer(
        dataset: LeRobotDataset,
        rb: ReplayBuffer,
        image_keys: list[str],
        num_episodes: int | None = None,
        use_base_policy_for_base_actions: bool = False,
        # base_policy: ACTPolicy | None = None,
        base_policy: ResidualClient | None = None,
    ) -> int:
        """
        Iterates through *dataset* sequentially, converts consecutive frames
        into residual RL transitions and pushes them into *rb*.

        Two modes:
        1. GT-as-base (use_base_policy_for_base_actions=False):
           Uses GT actions as both the base action (in observations) and the target action
           (in transitions). Teaches residual policy to output zero: residual = GT - GT = 0

        2. Base-policy-as-base (use_base_policy_for_base_actions=True):
           Uses base policy to generate base actions and GT actions as targets.
           More consistent with online training: residual = GT - base_policy_action

        Returns the number of transitions added.
        """
        if use_base_policy_for_base_actions and base_policy is None:
            raise ValueError("base_policy must be provided when use_base_policy_for_base_actions=True")

        # Populate buffer from pre-loaded dataset
        print("Populating offline buffer from dataset...")
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)   # todo define the offline data :  dataset

        episode_cache: dict[int, dict] = {}
        transitions = 0
        step_id = 0

        for sample in tqdm(loader, desc="Processing offline dataset"):  # 用进度条逐样本处理
            ep_idx = int(sample["episode_index"].item())
            if num_episodes is not None and ep_idx == num_episodes:
                break

            # ------------------------------------------------------------------
            # Build observation and action directly for replay buffer ----------
            # ------------------------------------------------------------------
            # Extract data and keep on CPU (replay buffer uses CPU storage)
            # _gt_action: torch.Tensor = sample["action"].float().squeeze(0)
            _gt_action: torch.Tensor = sample["eef_actions"].float().squeeze(0)
            gt_action_scaled = action_scaler.scale(_gt_action)
            # done_flag = bool(sample["next.done"].item())
            if "next.done" in sample:  # todo!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                done_flag = bool(sample["next.done"].item())
            elif "done" in sample:
                done_flag = bool(sample["done"].item())
            else:
                done_flag = False

            # Generate base action based on the selected mode
            if use_base_policy_for_base_actions:
                # Use base policy to generate base action from current observation
                # Build raw observation first for base policy inference
                raw_obs = {}
                for k in sample:
                    if "exterior_image_1_left" in k or "wrist_image_left" in k or "exterior_image_2_left" in k or "eef_position" in k or "gripper_position" in k:
                        raw_obs[k] = sample[k].to(device)  # Keep batch dimension for base policy

                # Get base action from base policy
                with torch.no_grad():
                    # base_action = base_policy.select_action(raw_obs)
                    # _, base_action, _, _, _ = base_policy.get_obs_and_base_action(raw_obs=raw_obs)  # todo done
                    base_action = base_policy.get_offline_action_base(raw_obs)  # todo done
                base_action_scaled = action_scaler.scale(base_action.squeeze(0).cpu())
            else:
                # Use GT action as base action (original behavior)
                base_action_scaled = gt_action_scaled

            # Build observation dict directly in target format'
            state = torch.cat((sample["eef_position"],sample["gripper_position"].unsqueeze(-1)),dim = -1)
            curr_obs = {
                "observation.state": state_standardizer.standardize(state.float().squeeze(0)),  # todo 
                "observation.base_action": base_action_scaled,
            }
            for k in image_keys:
                curr_obs[k] = sample[k].squeeze(0)
            curr_obs = process_image_batch(curr_obs, image_keys, enc_type, rb=True)  ###
            # Convert images to uint8 for memory-efficient storage
            to_uint8(curr_obs, image_keys)

            # ------------------------------------------------------------------
            # If we already cached the *previous* frame for this episode we can
            # create transitions now.
            # ------------------------------------------------------------------
            if ep_idx in episode_cache:
                # Create transitions for each combination of prev and current variants
                prev_obs = episode_cache[ep_idx]["obs"]
                # prev_obs = process_image_batch(prev_obs, image_keys, enc_type, rb=True)  ###
                prev_action_scaled = episode_cache[ep_idx]["action"]
                transition = TensorDict(
                    {
                        "obs": TensorDict(prev_obs, batch_size=[]),
                        "action": prev_action_scaled,
                        "next": TensorDict(
                            {
                                "obs": TensorDict(curr_obs, batch_size=[]),
                                "done": torch.tensor(done_flag, dtype=torch.bool),
                                "reward": torch.tensor(float(done_flag), dtype=torch.float32),
                            },
                            batch_size=[],
                        ),
                        "_priority": torch.tensor(10.0, dtype=torch.float32),  # High initial priority for new samples
                    },
                    batch_size=[],
                ).unsqueeze(0)

                rb.add(transition)
                transitions += 1

                step_id += 1
            else:
                step_id = 0

            # Cache current frame for pairing with the next one ---------------
            episode_cache[ep_idx] = {
                "obs": curr_obs,
                "action": gt_action_scaled,
                "done": done_flag,
                "step_id": step_id,
            }

        # Log final statistics
        print(f"Added {transitions} transitions")

        return transitions

    # ------------------------------------------------------------------
    # Caching layer for offline replay buffer ---------------------------
    # ------------------------------------------------------------------
    # Build a metadata dictionary that uniquely identifies the buffer
    offline_cache_meta = {
        "task": cfg.task,
        "dataset_name": cfg.offline_data.name,
        "num_episodes": cfg.offline_data.num_episodes,
        "use_base_policy_for_base_actions": cfg.offline_data.use_base_policy_for_base_actions,
        "min_action_range": cfg.offline_data.min_action_range,
        "min_state_std": cfg.offline_data.min_state_std,
        "image_keys": image_keys,
        "n_step": cfg.algo.n_step,
        "gamma": cfg.algo.gamma,
        "base_policy_wandb_id": cfg.base_policy.wandb_id,
        "sampling_strategy": cfg.algo.sampling_strategy,
        "normalized_actions": True,
        "batch_size": offline_batch_size,
        # Library versions for compatibility
        "torchrl_version": torchrl.__version__,
        "tensordict_version": tensordict.__version__,
    }
    if cfg.algo.sampling_strategy == "prioritized_replay":
        offline_cache_meta["priority_alpha"] = cfg.algo.priority_alpha
        offline_cache_meta["priority_beta"] = cfg.algo.priority_beta

    pprint.pprint(offline_cache_meta)

    # Deterministically hash the metadata to create a short cache directory name
    meta_str = json.dumps(offline_cache_meta, sort_keys=True)
    cache_hash = hashlib.sha1(meta_str.encode()).hexdigest()[:8]  # noqa: S324

    # Base local path for this buffer ---------------------------------------
    cache_dir = OFFLINE_CACHE_DIR / cache_hash

    # Try to download/extract from the Hub (will no-op if file not there)
    downloaded_dir = None
    if OFFLINE_HF_REPO is not None:
        print(f"Attempting to download offline buffer {cache_hash} from {OFFLINE_HF_REPO}...")
        downloaded_dir = _hf_download_buffer(OFFLINE_HF_REPO, cache_hash, OFFLINE_CACHE_DIR)
    if downloaded_dir is not None:
        cache_dir = downloaded_dir  # use extracted location

    loaded_from_cache = False
    added = 0

    if cfg.algo.offline_fraction > 0.0:
        # Only populate offline buffer if we're using offline data
        if cache_dir.exists():
            print(f"{cache_dir} found on disk. Attempting to load...")
            offline_rb.sampler._empty()
            optimized_replay_buffer_loads(offline_rb, cache_dir)
            loaded_from_cache = True
            print(f"Loaded offline buffer from cache at {cache_dir} (size={len(offline_rb)})")

        if not loaded_from_cache:
            added = _populate_offline_buffer(
                dataset=dataset,
                rb=offline_rb,
                image_keys=image_keys,
                num_episodes=cfg.offline_data.num_episodes,
                use_base_policy_for_base_actions=cfg.offline_data.use_base_policy_for_base_actions,
                base_policy=base_policy if cfg.offline_data.use_base_policy_for_base_actions else None,
            )

            print(f"Added {added} offline transitions to buffer (size={len(offline_rb)})")

            # Save buffer to disk for future runs + upload to Hub ----------------
            cache_dir.mkdir(parents=True, exist_ok=True)
            optimized_replay_buffer_dumps(offline_rb, cache_dir)

            with open(cache_dir / "user_metadata.json", "w") as f:
                json.dump(offline_cache_meta, f, indent=2)

            if OFFLINE_HF_REPO is not None:
                _hf_upload_buffer(OFFLINE_HF_REPO, cache_dir, cache_hash)
        else:
            added = len(offline_rb)
    else:
        print("Skipping offline buffer population for online-only training")
    
    # ------------------------------------------------------------------
    # Warm-up phase (random policy) --------------------------------------
    # ------------------------------------------------------------------

    if len(online_rb) < cfg.algo.learning_starts and not loaded_online_from_cache:
        print(f"Warm-up: filling online buffer with {cfg.algo.learning_starts - len(online_rb)} random steps…")
        # obs, _ = env.reset()
        obs = base_policy.reset() # todo reset
        # --------------------------------------------------------------
        # Logging helper: print progress every 1 000 collected transitions
        # --------------------------------------------------------------
        next_log_threshold = 1000  # first threshold for progress message

        reward_sum = 0
        episode_count = 0

        while len(online_rb) < cfg.algo.learning_starts:
            print(f"[warmup] len(online_rb) before step = {len(online_rb)}")
            if cfg.algo.use_base_policy_for_warmup:
                # Use base policy action + noise (residual exploration)
                # Since the environment wrapper always adds base_action to residual_action,
                # we just need to provide the noise as the residual action
                rand_actions = (  # line2: Sample noise εt ∼ U (−noise scale, noise scale)
                    torch.rand((cfg.num_envs, action_dim), device=device) * 2 - 1
                ) * cfg.algo.random_action_noise_scale
            else:
                # Pure uniform random actions - need to cancel out the base policy action
                # Since env does: combined = base_action + residual_action
                # To get pure random: residual_action = random - base_action
                # base_action = obs["observation.base_action"]  # Already normalized to [-1, 1]
                base_action = base_policy._last_base_action # todo need to normailize  ************may need to reset??

                pure_random = (
                    torch.rand((cfg.num_envs, action_dim), device=device) * 2 - 1
                ) * cfg.algo.random_action_noise_scale
                rand_actions = pure_random - base_action

            # line4: Observe next state st+1, reward rt, done flag dt
            # next_obs, reward, terminated, truncated, info = env.step(rand_actions)  # line3: Step env with at = εt + atb where atb ∼ πb(st)

            # next_obs, base_action, reward, terminated, truncated, info = base_policy.step(residual_action=rand_actions) # todo need to return  reward, terminated, truncated, info  normalize 
            # done = terminated | truncated
            next_obs, reward, done, info = base_policy.step(residual_action=rand_actions) # todo need to return  reward, terminated, truncated, info  normalize 
            # print(f"[warmup] after step: reward={reward}, done={done}")

            reward_sum += reward.sum().item()
            episode_count += done.float().sum().item()
            # reward_sum += reward
            # episode_count += int(done)

            # Use the executed combined action returned by the environment
            combined_action = info["scaled_action"]
            _add_transitions_to_buffer(  # line5: Add transition (st, atb, at, st+1, atb+1, rt, dt) to online replay buffer
                obs=obs,
                next_obs=next_obs,
                actions=combined_action,
                reward=reward,
                done=done,
                info=info,
                device=device,
                image_keys=image_keys,
                lowdim_keys=lowdim_keys,
                num_envs=cfg.num_envs,
                online_rb=online_rb,
                enc_type=enc_type
            )

            # ----------------------------------------------------------
            # Progress logging (every ~1 000 transitions) --------------
            # ----------------------------------------------------------
            if len(online_rb) >= next_log_threshold:  # todo
                success_rate = reward_sum / episode_count if episode_count > 0 else 0.0
                print(
                    f"[Warm-up] {len(online_rb)} / {cfg.algo.learning_starts} "
                    f"transitions collected, reward_sum={reward_sum:.2f}, "
                    f"success_rate={success_rate:.3f} ({reward_sum}/{episode_count})"
                )
                next_log_threshold += 1000

            obs = next_obs  # roll state

        # Persist freshly-collected buffer (local + HF) --------------------
        online_cache_dir.mkdir(parents=True, exist_ok=True)
        optimized_replay_buffer_dumps(online_rb, online_cache_dir)
        with open(online_cache_dir / "user_metadata.json", "w") as f:
            json.dump(online_cache_meta, f, indent=2)
        if ONLINE_HF_REPO is not None:
            _hf_upload_buffer(ONLINE_HF_REPO, online_cache_dir, online_cache_hash)
        print(f"Warm-up done. Online buffer size = {len(online_rb)} transitions")

        loaded_online_from_cache = True  # treat as cached going forward
    
    _hp_parts: list[str] = [
        cfg.task,  # e.g. "TwoArmBoxCleanup"
        f"n{cfg.algo.n_step}",  # n-step horizon
        f"utd{cfg.algo.num_updates_per_iteration}",  # updates-to-data ratio
        f"buf{cfg.algo.buffer_size}",  # replay buffer size
    ]

    # Offline dataset statistics (if any)
    if cfg.offline_data is not None and cfg.offline_data.num_episodes is not None and cfg.algo.offline_fraction > 0.0:
        _hp_parts.append(f"off{cfg.offline_data.num_episodes}ep")
    elif cfg.algo.offline_fraction == 0.0:
        _hp_parts.append("online_only")

    # Learning-rate, expressed in scientific notation for brevity (e.g. 1e-4 → 1e-04)
    _hp_parts.append(f"lr{cfg.agent.actor_lr:.0e}")

    # Additional flags ---------------------------------------------------------
    if cfg.agent.clip_q_target_to_reward_range:
        _hp_parts.append("clipT")

    hp_str = "_".join(_hp_parts)

    run_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}__{hp_str}__seed{cfg.seed}"

    if cfg.wandb.name is not None:
        run_name = f"{cfg.wandb.name}__{run_name}"

    _wandb_config = OmegaConf.to_container(cfg, resolve=True)
    # Remove notes from config if present
    assert isinstance(_wandb_config, dict)
    _wandb_config["wandb"].pop("notes", None)

    # Print a nice summary of the config
    print("Launching run with the following config:")
    pprint.pprint(_wandb_config)

    wandb.init(
        id=cfg.wandb.continue_run_id,
        resume=None if cfg.wandb.continue_run_id is None else "allow",
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=_wandb_config,
        name=run_name,
        mode=cfg.wandb.mode if not cfg.debug else "disabled",
        notes=cfg.wandb.notes,
        group=cfg.wandb.group,
    )

    # Log horizon to wandb summary
    # wandb.summary["environment/horizon"] = env.vec_env.metadata["horizon"]

    # Create a timestamped folder in CACHE_DIR for all outputs
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_cache_dir = _CACHE_ROOT / f"run_{timestamp}_{run_name}"

    # Create subdirectories for models and outputs
    model_save_dir = run_cache_dir / "models"
    outputs_dir = run_cache_dir / "outputs"
    model_save_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # obs, _ = env.reset()
    obs = base_policy.reset() # todo reset

    global_step = 0
    best_eval_success_rate = 0.0
    training_cum_time = 0.0
    episode_count = 0
    actor_updates = 0

    train_start_time = time.time()

    # Initialize timing utility
    training_timer = TrainingTimer()

    def _run_critic_warmup(
        agent, online_rb, offline_rb, cfg, device, training_timer, online_batch_size, offline_batch_size
    ):
        """Run critic-only updates for warmup phase."""
        # def print_leaf_shapes(td, name):
        #     print(f"\n===== {name} =====")
        #     for key, value in td.items(include_nested=True, leaves_only=True):
        #         if hasattr(value, "shape"):
        #             print(f"{key}: shape={tuple(value.shape)}, dtype={value.dtype}")
        #         else:
        #             print(f"{key}: type={type(value)}")

        # def compare_td_shapes(td1, td2, name1="online", name2="offline"):
        #     d1 = {}
        #     d2 = {}

        #     for key, value in td1.items(include_nested=True, leaves_only=True):
        #         d1[str(key)] = tuple(value.shape) if hasattr(value, "shape") else str(type(value))

        #     for key, value in td2.items(include_nested=True, leaves_only=True):
        #         d2[str(key)] = tuple(value.shape) if hasattr(value, "shape") else str(type(value))

        #     all_keys = sorted(set(d1.keys()) | set(d2.keys()))
        #     print("\n===== SHAPE MISMATCH CHECK =====")
        #     for k in all_keys:
        #         s1 = d1.get(k, None)
        #         s2 = d2.get(k, None)
        #         if s1 != s2:
        #             print(f"{k}: {name1}={s1}, {name2}={s2}")

        # for i in range(cfg.algo.critic_warmup_steps):
        #     with training_timer.time("batch_sampling"):
        #         online_batch = online_rb.sample(online_batch_size)
        #         online_batch = online_batch.to(device, non_blocking=True)

        #         if cfg.algo.offline_fraction > 0.0:
        #             offline_batch = offline_rb.sample(offline_batch_size)
        #             offline_batch = offline_batch.to(device, non_blocking=True)

        #             # 只在第一次 warmup 时打印，避免刷屏
        #             if i == 0:
        #                 print_leaf_shapes(online_batch, "ONLINE")
        #                 print_leaf_shapes(offline_batch, "OFFLINE")
        #                 compare_td_shapes(online_batch, offline_batch)

        #             batch = torch.cat([online_batch, offline_batch], dim=0)
        #         else:
        #             batch = online_batch
        for i in range(cfg.algo.critic_warmup_steps):
            # Sample mixed online/offline batch
            with training_timer.time("batch_sampling"):
                # Sample batches from replay buffers
                online_batch = online_rb.sample(online_batch_size)
                online_batch = online_batch.to(device, non_blocking=True)

                if cfg.algo.offline_fraction > 0.0:
                    # Mixed online/offline training
                    offline_batch = offline_rb.sample(offline_batch_size)
                    offline_batch = offline_batch.to(device, non_blocking=True)

                    batch = torch.cat([online_batch, offline_batch], dim=0)
                else:
                    # Online-only training
                    batch = online_batch

            # Only update critic during warmup (update_actor=False)
            with training_timer.time("gradient_update"):
                metrics = agent.update(batch, stddev=0.0, update_actor=False, bc_batch=None, ref_agent=agent)

            # Update priorities for prioritized experience replay
            if cfg.algo.sampling_strategy == "prioritized_replay" and "_td_errors" in metrics:
                # Update priorities in the batch for priority updates
                td_errors = metrics["_td_errors"]
                batch["_priority"] = td_errors

                if cfg.algo.offline_fraction > 0.0:
                    # Mixed online/offline training - update both buffers
                    online_batch_size_actual = int(cfg.algo.batch_size * (1 - cfg.algo.offline_fraction))

                    # Update online buffer priorities
                    if online_batch_size_actual > 0:
                        online_batch_subset = batch[:online_batch_size_actual]
                        online_rb.update_tensordict_priority(online_batch_subset)

                    # Update offline buffer priorities
                    if online_batch_size_actual < len(batch):
                        offline_batch_subset = batch[online_batch_size_actual:]
                        offline_rb.update_tensordict_priority(offline_batch_subset)
                else:
                    # Online-only training - update only online buffer
                    online_rb.update_tensordict_priority(batch)

            # Progress logging
            if i % 100 == 0:
                print(
                    f"Critic warmup: {i} / {cfg.algo.critic_warmup_steps}, "
                    f"train/critic_qt={metrics['train/critic_qt']:.4f} "
                    f"train/critic_loss={metrics['train/critic_loss']:.4f}"
                )

    if getattr(cfg, "resume", False) and getattr(cfg, "resume_checkpoint", None):
        print(f"Resuming training from checkpoint: {cfg.resume_checkpoint}")
        resume_state = load_training_checkpoint(
            ckpt_path=cfg.resume_checkpoint,
            agent=agent,
            online_rb=online_rb,
            offline_rb=offline_rb,
            device=device,
            load_replay_buffers=False,
        )

        global_step = resume_state["global_step"]
        best_eval_success_rate = resume_state["best_eval_success_rate"]
        training_cum_time = resume_state["training_cum_time"]
        episode_count = resume_state["episode_count"]
        actor_updates = resume_state["actor_updates"]

        # resume 后重新 reset env，继续采样
        obs = base_policy.reset()
    
    # ------------------------------------------------------------------
    # Critic warmup phase ----------------------------------------------
    # ------------------------------------------------------------------
    if cfg.algo.critic_warmup_steps > 0 and not cfg.resume:
        print(f"Critic warmup: running {cfg.algo.critic_warmup_steps} critic-only updates...")
        _run_critic_warmup(
            agent=agent,
            online_rb=online_rb,
            offline_rb=offline_rb,
            cfg=cfg,
            device=device,
            training_timer=training_timer,
            online_batch_size=online_batch_size,
            offline_batch_size=offline_batch_size,
        )
        print("Critic warmup completed.")

    while global_step <= cfg.algo.total_timesteps:
        iter_start = time.time()
        # ------------------------------------------------------------------
        # (1) Collect action + Environment step ---------------------------
        # ------------------------------------------------------------------
        with training_timer.time("env_step"):
            with torch.no_grad(), utils.eval_mode(agent):
                stddev = utils.schedule(cfg.algo.stddev_schedule, global_step)

                obs_act = process_image_batch(obs, image_keys, enc_type, rb=False)  ###
                action = agent.act(obs_act, eval_mode=False, stddev=stddev, cpu=False)  # line8: atr !!!

            if cfg.algo.progressive_clipping_steps > 0:
                clip_factor = min(1.0, global_step / cfg.algo.progressive_clipping_steps)
                action = action * clip_factor

            # next_obs, reward, terminated, truncated, info = env.step(action)  # line9: at(atr)
            # done = terminated | truncated  # terminated：任务本身的终止条件满足 truncated：被外部强制截断（时间上限等）
            next_obs, reward, done, info = base_policy.step(residual_action=action)

        if done.any():
            episode_count += done.float().sum().item()
            # # Extract episode information from final_info
            # final_info = info["final_info"]
            # episode_steps = final_info["episode_steps"]
            # episode_indices = final_info["_episode_steps"]

            # # Calculate discounted episode return
            # discount_factor = cfg.algo.gamma ** episode_steps[episode_indices]
            # episode_rewards = reward.cpu().numpy()[episode_indices]
            # episode_return = np.mean(discount_factor * episode_rewards)

            wandb.log(
                {
                    # "training/episode_return": episode_return,
                    # "training/episode_steps": episode_steps,
                    "training/episode_count": episode_count,
                },
                step=global_step,
            )

        # ------------------------------------------------------------------
        # (2) Add to online replay buffer ----------------------------------
        # ------------------------------------------------------------------
        # Use the executed combined action returned by the environment
        combined_action = info["scaled_action"]
        _add_transitions_to_buffer(  # line10: Add (st, atb, at, st+1, atb+1, rt, dt) to online replay buffer
            obs=obs,
            next_obs=next_obs,
            actions=combined_action,
            reward=reward,
            done=done,
            info=info,
            device=device,
            image_keys=image_keys,
            lowdim_keys=lowdim_keys,
            num_envs=cfg.num_envs,
            online_rb=online_rb,
            enc_type=enc_type
        )
        print("---------------------- _add_transitions_to_buffer ----------------------")

        obs = next_obs  # roll

        # ------------------------------------------------------------------
        # (3) Periodic evaluation ------------------------------------------
        # ------------------------------------------------------------------
        if global_step % cfg.eval_interval_every_steps == 0 and (cfg.eval_first or global_step > 0):
            with training_timer.time("evaluation"):
                eval_metrics = run_franka_evaluation(
                    env=base_policy,  ## todo eval?
                    agent=agent,
                    num_episodes=cfg.eval_num_episodes,
                    device=device,
                    global_step=global_step,
                    save_video=cfg.save_video,
                    save_q_plots=cfg.save_video,  # Enable Q-plots when video saving is enabled
                    run_name=run_name,
                    output_dir=outputs_dir,
                )

                # Handle model saving when success rate improves
                current_success_rate = eval_metrics["eval/success_rate"]
                if current_success_rate > best_eval_success_rate:
                    print(f"🎉 New best success rate: {current_success_rate:.4f} (prev: {best_eval_success_rate:.4f})")
                    best_eval_success_rate = current_success_rate

        global_step += cfg.num_envs

        # ------------------------------------------------------------------
        # (4) Updates -------------------------------------------------------
        # ------------------------------------------------------------------
        if global_step % cfg.algo.update_every_n_steps == 0 or global_step == cfg.num_envs:  # 不是每一步 env step 都更新网络，而是攒够 n 步再更新一次
            i = 0
            actor_update_cadence = cfg.algo.num_updates_per_iteration // cfg.algo.actor_updates_per_iteration
            # Normal training loop - critic is already warmed up
            while i < cfg.algo.num_updates_per_iteration:
                # --------------------------------------------------------------
                # Sample mixed online/offline batch
                # --------------------------------------------------------------
                with training_timer.time("batch_sampling"):
                    # Sample batches from replay buffers
                    online_batch = online_rb.sample(online_batch_size)
                    online_batch = online_batch.to(device, non_blocking=True)

                    if cfg.algo.offline_fraction > 0.0:  # offline_fraction 决定 offline 占 batch 的多少
                        # Mixed online/offline training
                        offline_batch = offline_rb.sample(offline_batch_size)
                        offline_batch = offline_batch.to(device, non_blocking=True)
                        batch = torch.cat([online_batch, offline_batch], dim=0)
                    else:
                        # Online-only training
                        batch = online_batch

                # Update actor on the last iteration of each update cycle
                update_actor = (i + 1) % actor_update_cadence == 0

                # Apply actor learning rate warmup
                if update_actor:
                    if cfg.algo.actor_lr_warmup_steps > 0:
                        # Calculate current LR with linear warmup from 0 to target
                        warmup_progress = min(1.0, actor_updates / cfg.algo.actor_lr_warmup_steps)
                        current_lr = cfg.agent.actor_lr * warmup_progress
                        for param_group in agent.actor_opt.param_groups:
                            param_group["lr"] = current_lr

                    actor_updates += 1

                with training_timer.time("gradient_update"):
                    metrics = agent.update(batch, stddev, update_actor, bc_batch=None, ref_agent=agent)

                # Update priorities for prioritized experience replay
                if cfg.algo.sampling_strategy == "prioritized_replay" and "_td_errors" in metrics:
                    # Update priorities in the batch for priority updates
                    td_errors = metrics["_td_errors"]
                    batch["_priority"] = td_errors

                    if cfg.algo.offline_fraction > 0.0:
                        # Mixed online/offline training - update both buffers
                        online_batch_size_actual = int(cfg.algo.batch_size * (1 - cfg.algo.offline_fraction))

                        # Update online buffer priorities
                        if online_batch_size_actual > 0:
                            online_batch_subset = batch[:online_batch_size_actual]
                            online_rb.update_tensordict_priority(online_batch_subset)

                        # Update offline buffer priorities
                        if online_batch_size_actual < len(batch):
                            offline_batch_subset = batch[online_batch_size_actual:]
                            offline_rb.update_tensordict_priority(offline_batch_subset)
                    else:
                        # Online-only training - update only online buffer
                        online_rb.update_tensordict_priority(batch)

                metrics["data/batch_terminal_R"] = batch["next"]["reward"][~batch["nonterminal"]].mean()
                metrics["data/terminal_share"] = (~batch["nonterminal"]).float().mean()

                i += 1

        training_cum_time += time.time() - iter_start

        # ------------------------------------------------------------------
        # (6) Logging -------------------------------------------------------
        # ------------------------------------------------------------------
        if global_step % cfg.log_freq == 0:
            sps = int(global_step / training_cum_time) if training_cum_time > 0 else 0

            # Prepare base logging dict
            log_dict = {
                "training/SPS": sps,
                "training/global_step": global_step,
                "buffer/online_size": len(online_rb),
                "buffer/offline_size": len(offline_rb) if offline_rb else 0,
                "timing/training_total_time": time.time() - train_start_time,
                "timing/aggregate_steps_per_second": global_step / (time.time() - train_start_time),
                "training/actor_lr": agent.actor_opt.param_groups[0]["lr"],
            }

            # Add timing statistics
            timing_stats = training_timer.get_timing_stats()
            log_dict.update(timing_stats)

            # Add metrics, filtering out internal data
            filtered_metrics = {k: v for k, v in metrics.items() if not k.startswith("_")}
            log_dict.update(filtered_metrics)

            # Compute residual action statistics only when logging
            if "_actions" in metrics:
                actions = metrics["_actions"]
                # Compute L1/L2 magnitudes (only during logging to save computation)
                residual_l1_magnitude = torch.mean(torch.abs(actions)).item()
                residual_l2_magnitude = torch.mean(torch.square(actions)).item()

                log_dict["train/residual_l1_magnitude"] = residual_l1_magnitude
                log_dict["train/residual_l2_magnitude"] = residual_l2_magnitude
                log_dict["histograms/residual_actions"] = wandb.Histogram(actions.numpy().reshape(-1))
            else:
                residual_l1_magnitude = None
                residual_l2_magnitude = None

            # Add Q values histogram when available
            if "_target_q" in metrics:
                target_q = metrics["_target_q"]
                log_dict["histograms/critic_qt"] = wandb.Histogram(target_q.numpy().reshape(-1))

            if cfg.algo.progressive_clipping_steps > 0:
                log_dict["training/progressive_clipping_factor"] = clip_factor

            wandb.log(log_dict, step=global_step)

            # Enhanced print statement with residual action magnitudes, gradient norms, and actor LR
            current_actor_lr = agent.actor_opt.param_groups[0]["lr"]

            if "train/actor_loss_base" in metrics:
                actor_loss_str = f"actor_loss_base={metrics['train/actor_loss_base']:.4f}"
                print_msg = (
                    f"[{global_step}] {actor_loss_str} "
                    f"critic_loss={metrics['train/critic_loss']:.4f} "
                    f"actor_lr={current_actor_lr:.2e}"
                )
            else:
                # During critic warmup, actor might not be updated
                print_msg = (
                    f"[{global_step}] critic_loss={metrics['train/critic_loss']:.4f} "
                    f"actor_lr={current_actor_lr:.2e} (actor not updated)"
                )
            if residual_l1_magnitude is not None and residual_l2_magnitude is not None:
                print_msg += f" residual_l1={residual_l1_magnitude:.4f} residual_l2={residual_l2_magnitude:.4f}"

            # Add gradient norms to print statement
            if "train/actor_grad_norm" in metrics:
                print_msg += f" actor_grad_norm={metrics['train/actor_grad_norm']:.4f}"

            # Add L2 penalty if active
            if "train/actor_l2_penalty" in metrics:
                print_msg += f" l2_penalty={metrics['train/actor_l2_penalty']:.4f}"

            # Add timing percentages to print statement
            if timing_stats:
                env_pct = timing_stats.get("timing/env_step_percentage", 0)
                grad_pct = timing_stats.get("timing/gradient_update_percentage", 0)
                batch_pct = timing_stats.get("timing/batch_sampling_percentage", 0)
                eval_pct = timing_stats.get("timing/evaluation_percentage", 0)
                print_msg += (
                    f" | Time%: env={env_pct:.1f} grad={grad_pct:.1f} batch={batch_pct:.1f} eval={eval_pct:.1f}"
                )

            print(print_msg)
        if global_step % cfg.checkpoint_interval == 0:
            ckpt_path = model_save_dir / f"checkpoint_{global_step}.pt"
            save_training_checkpoint(
                ckpt_path=ckpt_path,
                agent=agent,
                online_rb=online_rb,
                offline_rb=offline_rb,
                global_step=global_step,
                best_eval_success_rate=best_eval_success_rate,
                training_cum_time=training_cum_time,
                episode_count=episode_count,
                actor_updates=actor_updates,
                run_name=run_name,
                cfg=cfg,
                save_replay_buffers=cfg.save_replay_on_checkpoint,
            )
    
    print(f"Training finished in {time.time() - train_start_time:.2f} seconds.")

    # Clean up entire run directory after successful completion (videos/logs are saved to wandb)
    if run_cache_dir.exists():
        print(f"Cleaning up run directory: {run_cache_dir}")
        shutil.rmtree(run_cache_dir)
        print("Run directory cleaned up successfully.")


# -----------------------------------------------------------------------------
# Hydra entry point -----------------------------------------------------------
# -----------------------------------------------------------------------------
@hydra.main(version_base=None, config_name="residual_td3_dexmg_config")
def hydra_entry(cfg: ResidualTD3DexmgConfig):
    cfg_conf = OmegaConf.structured(cfg)
    main(cfg_conf)


if __name__ == "__main__":
    hydra_entry()
