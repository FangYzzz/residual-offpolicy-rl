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
import queue
import copy
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
import threading
import queue
import wandb
from resfit.dexmg.environments.dexmg import create_vectorized_env
from resfit.lerobot.policies.act.configuration_act import ACTConfig
from resfit.lerobot.policies.act.modeling_act import ACTPolicy
from resfit.lerobot.utils.load_policy import download_policy_from_wandb, load_policy
from resfit.rl_finetuning.config.residual_td3 import ResidualTD3DexmgConfig
from resfit.rl_finetuning.off_policy.common_utils import utils
from resfit.rl_finetuning.off_policy.rl.q_agent_lang import QAgentLang
from resfit.rl_finetuning.utils.dtype import to_uint8
from resfit.rl_finetuning.utils.evaluate_dexmg import run_dexmg_evaluation
from resfit.rl_finetuning.utils.evaluate_franka import run_franka_evaluation
from resfit.rl_finetuning.utils.hugging_face import (
    _hf_download_buffer,
    _hf_upload_buffer,
    optimized_replay_buffer_dumps,
    optimized_replay_buffer_loads,
)
from resfit.rl_finetuning.config.rlpd import (
    LanguageConfig,
)
from resfit.rl_finetuning.utils.normalization import ActionScaler, StateStandardizer
from resfit.rl_finetuning.utils.rb_transforms import MultiStepTransform
from resfit.rl_finetuning.wrappers.residual_env_wrapper import BasePolicyVecEnvWrapper
from gpt_residual_robot import BasePolicy

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
# Language embedding helper ---------------------------------------------------
# -----------------------------------------------------------------------------
class LanguageEmbedder:
    """Compute a fixed sentence embedding for a task / prompt string.

    Frozen, lazy-loaded; the result is cached per string so we never re-run
    the LM during data collection.

    Default backend: ``sentence-transformers/all-MiniLM-L6-v2`` -> 384-dim
    embedding.  Swap ``model_name`` for any other HuggingFace sentence /
    text encoder; just keep ``cfg.agent.language.lang_emb_dim`` in sync.

    If the ``sentence-transformers`` package is missing, falls back to a
    deterministic hashed pseudo-embedding so the rest of the training
    pipeline still runs (useful for plumbing checks).
    """

    def __init__(
        self,
        *,
        emb_dim: int,
        device: torch.device,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.emb_dim = emb_dim
        self.device = device
        self.model_name = model_name
        self._cache: dict[str, torch.Tensor] = {}
        self._model = None
        self._loaded = False

    def _maybe_load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=str(self.device))
            self._model.eval()
            actual_dim = int(self._model.get_sentence_embedding_dimension())
            assert actual_dim == self.emb_dim, (
                f"LanguageEmbedder: model '{self.model_name}' produces {actual_dim}-dim "
                f"embeddings but cfg.agent.language.lang_emb_dim={self.emb_dim}. "
                f"Either change the model or update the config."
            )
        except Exception as e:  # noqa: BLE001
            print(
                f"⚠️  LanguageEmbedder: could not load '{self.model_name}' ({e}); "
                f"falling back to deterministic hashed pseudo-embeddings. "
                f"Install sentence-transformers for real embeddings."
            )
            self._model = None

    def encode(self, text: str) -> torch.Tensor:
        if text in self._cache:
            return self._cache[text]
        self._maybe_load()
        if self._model is not None:
            with torch.no_grad():
                vec = self._model.encode(
                    text,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                ).to(self.device).float()
        else:
            seed = abs(hash(text)) % (2**31 - 1)
            g = torch.Generator(device="cpu").manual_seed(seed)
            vec = torch.randn(self.emb_dim, generator=g)
            vec = (vec / vec.norm()).to(self.device).float()
        self._cache[text] = vec
        return vec

    def __call__(self, text: str) -> torch.Tensor:
        return self.encode(text)

def _load_lerobot_task_prompts(dataset: LeRobotDataset, dataset_name: str) -> dict[int, str]:
    """Read ``meta/tasks.jsonl`` and return task_index -> prompt."""
    candidate_paths: list[Path] = []

    dataset_root = getattr(dataset, "root", None)
    if dataset_root is not None:
        candidate_paths.append(Path(dataset_root) / "meta" / "tasks.jsonl")

    dataset_meta_root = getattr(getattr(dataset, "meta", None), "root", None)
    if dataset_meta_root is not None:
        candidate_paths.append(Path(dataset_meta_root) / "tasks.jsonl")
        candidate_paths.append(Path(dataset_meta_root) / "meta" / "tasks.jsonl")

    name_path = Path(dataset_name).expanduser()
    candidate_paths.append(name_path / "meta" / "tasks.jsonl")
    candidate_paths.append(name_path / "tasks.jsonl")

    tasks_path = next((path for path in candidate_paths if path.exists()), None)
    if tasks_path is None:
        searched = "\n  ".join(str(path) for path in candidate_paths)
        raise FileNotFoundError(f"Could not find LeRobot tasks.jsonl. Searched:\n  {searched}")

    task_prompts: dict[int, str] = {}
    with tasks_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "task_index" not in row or "task" not in row:
                raise KeyError(f"{tasks_path}:{line_no} must contain 'task_index' and 'task'")
            task_prompts[int(row["task_index"])] = str(row["task"])

    if not task_prompts:
        raise ValueError(f"No task prompts found in {tasks_path}")

    print(f"Loaded {len(task_prompts)} task prompts from {tasks_path}")
    return task_prompts

def _inject_task_emb(
    obs: dict[str, torch.Tensor], task_emb: torch.Tensor, key: str = "observation.task_emb"
) -> dict[str, torch.Tensor]:
    """Set ``obs[key]`` to ``task_emb`` (broadcast to whatever batch dim the
    rest of obs has).  Mutates ``obs`` in place and returns it for chaining.
    """
    # print("type(obs)::::::::::::::::::", type(obs))
    if key in obs:
        return obs
    state = obs.get("observation.state", None)
    if state is not None and state.dim() == 2:
        emb = task_emb.unsqueeze(0).expand(state.size(0), -1).contiguous()
    else:
        emb = task_emb
    obs[key] = emb.to(state.device if state is not None else task_emb.device)
    return obs


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
    out_size = 224 if enc_type == "siglip" else 84  # cfg.agent.enc_type == "vit"  # !!!
    # out_size = 84
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
    global_step: int,
    best_eval_success_rate: float,
    training_cum_time: float,
    episode_count: int,
    actor_updates: int,
    run_name: str,
    cfg,
):
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "global_step": global_step,
        # Full agent state_dict captures every registered submodule:
        # encoders, actor, critic, actor_target, critic_target (and any new
        # submodule added later, e.g. lang_encoder).  This is the key fix
        # for the "load -> success_rate=0" bug: previously `encoders` was
        # never saved, so after loading we evaluated trained actor/critic
        # against a *freshly initialized* image encoder.
        "agent": agent.state_dict(),

        # Per-component dicts kept for backwards compatibility with eval
        # scripts that read them directly.
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "encoders": agent.encoders.state_dict(),
        "lang_encoder": agent.lang_encoder.state_dict(),
        "actor_target": agent.actor_target.state_dict(),
        "critic_target": agent.critic_target.state_dict(),

        # All optimizers (encoder_opt was previously missing).
        "actor_opt": agent.actor_opt.state_dict(),
        "critic_opt": agent.critic_opt.state_dict(),
        "encoder_opt": agent.encoder_opt.state_dict(),

        # LR schedulers (None-safe).
        "encoder_scheduler": (
            agent.encoder_scheduler.state_dict() if agent.encoder_scheduler is not None else None
        ),
        "critic_scheduler": (
            agent.critic_scheduler.state_dict() if agent.critic_scheduler is not None else None
        ),
        "actor_scheduler": (
            agent.actor_scheduler.state_dict() if agent.actor_scheduler is not None else None
        ),

        "best_eval_success_rate": best_eval_success_rate,
        "training_cum_time": training_cum_time,
        "episode_count": episode_count,
        "actor_updates": actor_updates,
        "run_name": run_name,
        "wandb_run_id": wandb.run.id if wandb.run is not None else None,
        "cfg": OmegaConf.to_container(cfg, resolve=True),

        # RNG states
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }

    torch.save(checkpoint, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    



def load_training_checkpoint(
    ckpt_path: str | Path,
    agent,
    device: torch.device,
):
    ckpt_path = Path(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # ---------- Networks ---------------------------------------------------
    # Prefer the full-agent state_dict if present (new format); this restores
    # encoders, actor, critic, actor_target, critic_target and any future
    # submodule in one shot.  Fall back to the old per-component layout for
    # backwards compatibility with checkpoints that pre-date this fix.
    loaded_modules: list[str] = []
    if "agent" in checkpoint:
        missing, unexpected = agent.load_state_dict(checkpoint["agent"], strict=False)
        loaded_modules.append("agent (full)")
        if missing:
            print(f"  [load] missing keys (will stay at __init__ values): {len(missing)}")
        if unexpected:
            print(f"  [load] unexpected keys (ignored): {len(unexpected)}")
    else:
        agent.actor.load_state_dict(checkpoint["actor"])
        loaded_modules.append("actor")
        agent.critic.load_state_dict(checkpoint["critic"])
        loaded_modules.append("critic")
        if "encoders" in checkpoint:
            agent.encoders.load_state_dict(checkpoint["encoders"])
            loaded_modules.append("encoders")
        else:
            print(
                "  ⚠️  Old checkpoint format: 'encoders' not found.  The image "
                "encoder will stay at __init__ (random) values, which is the "
                "exact reason eval success rate drops to 0%.  Re-train or "
                "continue training and re-save."
            )
        if "actor_target" in checkpoint:
            agent.actor_target.load_state_dict(checkpoint["actor_target"])
            loaded_modules.append("actor_target")
        else:
            agent.actor_target.load_state_dict(agent.actor.state_dict())
            loaded_modules.append("actor_target<-actor")
        if "critic_target" in checkpoint:
            agent.critic_target.load_state_dict(checkpoint["critic_target"])
            loaded_modules.append("critic_target")
        else:
            agent.critic_target.load_state_dict(agent.critic.state_dict())
            loaded_modules.append("critic_target<-critic")

    # ---------- Optimizers (continue-training only; safe to skip for eval) -
    if "actor_opt" in checkpoint:
        agent.actor_opt.load_state_dict(checkpoint["actor_opt"])
    if "critic_opt" in checkpoint:
        agent.critic_opt.load_state_dict(checkpoint["critic_opt"])
    if "encoder_opt" in checkpoint:
        agent.encoder_opt.load_state_dict(checkpoint["encoder_opt"])

    # ---------- LR schedulers ---------------------------------------------
    for name in ("encoder_scheduler", "critic_scheduler", "actor_scheduler"):
        sched = getattr(agent, name, None)
        if sched is not None and checkpoint.get(name) is not None:
            sched.load_state_dict(checkpoint[name])

    print(f"  [load] restored: {', '.join(loaded_modules)}")

    global_step = checkpoint.get("global_step", 0)
    best_eval_success_rate = checkpoint.get("best_eval_success_rate", 0.0)
    training_cum_time = checkpoint.get("training_cum_time", 0.0)
    episode_count = checkpoint.get("episode_count", 0)
    actor_updates = checkpoint.get("actor_updates", 0)
    run_name = checkpoint.get("run_name", None)
    wandb_run_id = checkpoint.get("wandb_run_id", None)

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

    
    print(f"Loaded checkpoint from {ckpt_path}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    return {
        "global_step": global_step,
        "best_eval_success_rate": best_eval_success_rate,
        "training_cum_time": training_cum_time,
        "episode_count": episode_count,
        "actor_updates": actor_updates,
        "run_name": run_name,
        "wandb_run_id": wandb_run_id,
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


def collector_loop(
        cfg,
        lang_cfg,
        lang_embedder,
        device,
        base_policy,
        image_keys,
        enc_type,
        episode_queue,
        weights_queue,
        stop_event,
        training_timer,
        run_name,
        outputs_dir,
        img_c,
        img_h,
        img_w,
        lowdim_dim,
        action_dim,
        initial_global_step=0,
        initial_episode_count=0,
        initial_best_eval_success_rate=-float("inf"),
    ):
        global_step = initial_global_step
        episode_count = initial_episode_count
        episode_idx = int(initial_episode_count)
        best_eval_success_rate = initial_best_eval_success_rate

        # obs = base_policy.reset()
        def _attach_task_emb(obs, task_emb):
            if not lang_cfg.enabled or task_emb is None or obs is None:
                return obs
            return _inject_task_emb(obs, task_emb, key=lang_cfg.lang_emb_obs_key)
        obs, task_prompt = base_policy.reset()
        task_emb = lang_embedder(task_prompt)
        obs = _attach_task_emb(obs, task_emb)
        agent = QAgentLang(obs_shape=(img_c, img_h, img_w),
                            prop_shape=(lowdim_dim,),
                            action_dim=action_dim,
                            rl_cameras=image_keys,
                            cfg=cfg.agent,
                            residual_actor=True,  # Enable residual actor mode
                        ) 
        def _try_load_latest_weights():
            latest = None
            while True:
                try:
                    latest = weights_queue.get_nowait()
                except queue.Empty:
                    break

            if latest is not None:
                agent.actor.load_state_dict(latest["actor"])
                agent.encoders.load_state_dict(latest["encoders"])
                agent.critic.load_state_dict(latest["critic"])
                agent.lang_encoder.load_state_dict(latest["lang_encoder"])
                # print("[collector] rollout actor updated #########################" )


        while (global_step <= cfg.algo.total_timesteps) and (not stop_event.is_set()):
            # 每个 episode 开始前尝试拿一次最新权重
            _try_load_latest_weights()

            episode_steps = []
            episode_done = False
            
            while not episode_done and global_step <= cfg.algo.total_timesteps and (not stop_event.is_set()) and len(episode_steps)<cfg.send_transitions_len:
                with torch.no_grad(), utils.eval_mode(agent):
                    # print("cfg.algo.stddev_schedule:", cfg.algo.stddev_schedule)  # 0.003
                    stddev = utils.schedule(cfg.algo.stddev_schedule, global_step)
                    # print("stddev:", stddev)  # 0.0030000000000000005
                    obs_act = process_image_batch(copy.deepcopy(obs), image_keys, enc_type, rb=False)
                    action = agent.act(obs_act, eval_mode=False, stddev=stddev, cpu=False)

                if cfg.algo.progressive_clipping_steps > 0:
                    clip_factor = min(1.0, global_step / cfg.algo.progressive_clipping_steps)  # 训练一开始不要让 residual action 太大，而是慢慢放开
                    action = action * clip_factor  # 让 residual policy 在训练早期影响较小，避免一开始破坏 base policy

                next_obs, reward, done, info = base_policy.step(residual_action=action)
                if done:
                    task_prompt = info["task_prompt"]
                    task_emb = lang_embedder(task_prompt)
                next_obs = _attach_task_emb(next_obs, task_emb)
                if done.any():
                    episode_count += done.float().sum().item()
                    episode_done = True
                    wandb.log(
                        {
                            "training/reward": reward
                        },
                        step= global_step,
                    )

                # 注意：这里先不 add_to_buffer，而是存起来（仍然是一步）
                step_payload = {
                    "obs": obs,
                    "next_obs": next_obs,
                    "action": info["scaled_action"],  # combined action
                    "reward": reward,
                    "done": done,
                    "info": info,
                }
                episode_steps.append(step_payload)  # append 一步
                obs = next_obs

            # 每收集一步就发给 learner
            ep_payload = {
                "episode_idx": episode_idx,
                "global_step_after_episode": global_step,
                "episode_count": episode_count,
                "steps": episode_steps,
            }

            
            episode_queue.put(ep_payload)
            print(f"[collector] pushed episode {episode_idx}, len={len(episode_steps)}, global_step={global_step}")

            episode_idx += 1
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
                        lang_cfg = lang_cfg,
                        lang_embedder= lang_embedder,
                    )

                    # Handle model saving when success rate improves
                    current_success_rate_task_one = eval_metrics["eval/success_rate_task_one"]
                    current_success_rate_task_two = eval_metrics["eval/success_rate_task_two"]
                    # if current_success_rate > best_eval_success_rate:
                    print(f"🎉 task one success rate: {current_success_rate_task_one}")
                    print(f"🎉 task two success rate: {current_success_rate_task_two}")
                        # best_eval_success_rate = current_success_rate
                obs, task_prompt = base_policy.reset()
                task_emb = lang_embedder(task_prompt)
                obs = _attach_task_emb(obs, task_emb)
            # 原来 eval 后会 reset，这里 episode 结束也 reset
            # obs = base_policy.reset()
            
            global_step += cfg.send_transitions_len
        stop_event.set()
        print("[collector] finished")


def learner_loop(
        cfg,
        device,
        agent,
        online_rb,
        offline_rb,
        image_keys,
        lowdim_keys,
        enc_type,
        episode_queue,
        weights_queue,
        stop_event,
        model_save_dir,
        run_name,
        online_cache_dir,
        online_cache_meta,
        online_cache_hash,
        initial_global_step=0,
        initial_actor_updates=0,
        initial_episode_count=0,
        initial_best_eval_success_rate=0.0,
        initial_training_cum_time=0.0,

    ):
        training_timer = TrainingTimer()

        global_step = initial_global_step
        actor_updates = initial_actor_updates
        episode_count = initial_episode_count
        best_eval_success_rate = initial_best_eval_success_rate
        training_cum_time = initial_training_cum_time
        train_start_time = time.time()

        metrics = {}

        online_batch_size = int(cfg.algo.batch_size * (1 - cfg.algo.offline_fraction))
        offline_batch_size = int(cfg.algo.batch_size * cfg.algo.offline_fraction)
        next_persist_threshold = len(online_rb) + cfg.save_online_rb_interval
        
        # 一开始把初始 actor 发给 collector
        weights_queue.put({
            "actor": {k: v.detach().cpu() for k, v in agent.actor.state_dict().items()},
            "encoders": {k: v.detach().cpu() for k, v in agent.encoders.state_dict().items()},
            "critic": {k: v.detach().cpu() for k, v in agent.critic.state_dict().items()},
            "lang_encoder": {k: v.detach().cpu() for k, v in agent.lang_encoder.state_dict().items()}
        })

        def _publish_latest_actor():
            payload = {
                "actor": {k: v.detach().cpu() for k, v in agent.actor.state_dict().items()},
                "encoders": {k: v.detach().cpu() for k, v in agent.encoders.state_dict().items()},
                "critic": {k: v.detach().cpu() for k, v in agent.critic.state_dict().items()},
                "lang_encoder": {k: v.detach().cpu() for k, v in agent.lang_encoder.state_dict().items()}
            }
            # 保持 queue 里尽量只有最新
            try:
                while True:
                    weights_queue.get_nowait()
            except Exception:
                pass
            weights_queue.put(payload)
        
        len_rb = len(online_rb)
        while (not stop_event.is_set()) or (not episode_queue.empty()):
            iter_start = time.time()

            # --------------------------------------------------------------
            # (A) 接收 collector 发来的 episode
            # --------------------------------------------------------------
            got_episode = False
            try:
                ep_payload = episode_queue.get_nowait()
            except queue.Empty:
                time.sleep(0.01)
                continue

            got_episode = True
            episode_count = ep_payload["episode_count"]
            global_step = max(global_step, ep_payload["global_step_after_episode"])

            steps = ep_payload["steps"]
            for st in steps:
                _add_transitions_to_buffer(
                    obs=st["obs"],
                    next_obs=st["next_obs"],
                    actions=st["action"].to(device),
                    reward=st["reward"].to(device),
                    done=st["done"].to(device),
                    info=st["info"],
                    device=device,
                    image_keys=image_keys,
                    lowdim_keys=lowdim_keys,
                    num_envs=cfg.num_envs,
                    online_rb=online_rb,
                    enc_type=enc_type,
                )
                len_rb+=1

            # print(f"next_persist_threshold: {next_persist_threshold} !!!!!!!!!!!!!!!!!!!! ")
            # print(f"len of online rb:   {len_rb} !!!!!!!!!!!!!")
            if len_rb >= next_persist_threshold:
                online_cache_dir.mkdir(parents=True, exist_ok=True)
                optimized_replay_buffer_dumps(online_rb, online_cache_dir)
                with open(online_cache_dir / "user_metadata.json", "w") as f:
                    json.dump(online_cache_meta, f, indent=2)
                if ONLINE_HF_REPO is not None:
                    _hf_upload_buffer(ONLINE_HF_REPO, online_cache_dir, online_cache_hash)
                print(f"Update online rb. Online buffer size = {len(online_rb)} transitions")

                if ONLINE_HF_REPO is not None:
                    _hf_upload_buffer(ONLINE_HF_REPO, online_cache_dir, online_cache_hash)

                next_persist_threshold += cfg.save_online_rb_interval

            # print(f"[learner] received episode {ep_payload['episode_idx']}, rb_size={len(online_rb)}")

            # # buffer 不够就先等
            # if len(online_rb) < cfg.algo.learning_starts:
            #     if not got_episode:
            #         time.sleep(0.01)
            #     continue
            if not got_episode:
                time.sleep(0.01)
                continue
            # --------------------------------------------------------------
            # (B) Updates
            # --------------------------------------------------------------
            if global_step % cfg.algo.update_every_n_steps == 0 or global_step == cfg.num_envs:
                # print("start updating !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                i = 0
                actor_update_cadence = cfg.algo.num_updates_per_iteration // cfg.algo.actor_updates_per_iteration

                while i < cfg.algo.num_updates_per_iteration:
                    with training_timer.time("batch_sampling"):
                        online_batch = online_rb.sample(online_batch_size)
                        online_batch = online_batch.to(device, non_blocking=True)

                        if cfg.algo.offline_fraction > 0.0:
                            offline_batch = offline_rb.sample(offline_batch_size)
                            offline_batch = offline_batch.to(device, non_blocking=True)
                            batch = torch.cat([online_batch, offline_batch], dim=0)
                        else:
                            batch = online_batch

                    update_actor = (i + 1) % actor_update_cadence == 0

                    stddev = utils.schedule(cfg.algo.stddev_schedule, global_step)

                    if update_actor:
                        if cfg.algo.actor_lr_warmup_steps > 0:
                            warmup_progress = min(1.0, actor_updates / cfg.algo.actor_lr_warmup_steps)
                            current_lr = cfg.agent.actor_lr * warmup_progress
                            for param_group in agent.actor_opt.param_groups:
                                param_group["lr"] = current_lr
                        actor_updates += 1

                    with training_timer.time("gradient_update"):
                        metrics = agent.update(batch, stddev, update_actor, bc_batch=None, ref_agent=agent)

                    if cfg.algo.sampling_strategy == "prioritized_replay" and "_td_errors" in metrics:
                        td_errors = metrics["_td_errors"]
                        batch["_priority"] = td_errors

                        if cfg.algo.offline_fraction > 0.0:
                            online_batch_size_actual = int(cfg.algo.batch_size * (1 - cfg.algo.offline_fraction))

                            if online_batch_size_actual > 0:
                                online_batch_subset = batch[:online_batch_size_actual]
                                online_rb.update_tensordict_priority(online_batch_subset)

                            if online_batch_size_actual < len(batch):
                                offline_batch_subset = batch[online_batch_size_actual:]
                                offline_rb.update_tensordict_priority(offline_batch_subset)
                        else:
                            online_rb.update_tensordict_priority(batch)

                    metrics["data/batch_terminal_R"] = batch["next"]["reward"][~batch["nonterminal"]].mean()
                    metrics["data/terminal_share"] = (~batch["nonterminal"]).float().mean()
                    i += 1

                # 每轮更新后发一次最新 actor 给 collector
            _publish_latest_actor()

            training_cum_time += time.time() - iter_start

            # --------------------------------------------------------------
            # (C) Logging
            # --------------------------------------------------------------
            if global_step > 0 and global_step % cfg.log_freq == 0 and metrics:
                sps = int(global_step / training_cum_time) if training_cum_time > 0 else 0

                log_dict = {
                    "training/SPS": sps,
                    "training/global_step": global_step,
                    "training/episode_count": episode_count,
                    "buffer/online_size": len(online_rb),
                    "buffer/offline_size": len(offline_rb) if offline_rb else 0,
                    "timing/training_total_time": time.time() - train_start_time,
                    "timing/aggregate_steps_per_second": global_step / (time.time() - train_start_time),
                    "training/actor_lr": agent.actor_opt.param_groups[0]["lr"],
                }

                timing_stats = training_timer.get_timing_stats()
                log_dict.update(timing_stats)

                filtered_metrics = {k: v for k, v in metrics.items() if not k.startswith("_")}
                log_dict.update(filtered_metrics)

                if "_actions" in metrics:
                    actions = metrics["_actions"]
                    residual_l1_magnitude = torch.mean(torch.abs(actions)).item()
                    residual_l2_magnitude = torch.mean(torch.square(actions)).item()

                    log_dict["train/residual_l1_magnitude"] = residual_l1_magnitude
                    log_dict["train/residual_l2_magnitude"] = residual_l2_magnitude
                    log_dict["histograms/residual_actions"] = wandb.Histogram(actions.numpy().reshape(-1))

                if "_target_q" in metrics:
                    target_q = metrics["_target_q"]
                    log_dict["histograms/critic_qt"] = wandb.Histogram(target_q.numpy().reshape(-1))
                
                last_sigma = getattr(agent.actor, "last_sigma", None)
                if last_sigma is not None:
                    log_dict["histograms/scale_sigma"] = wandb.Histogram(
                        last_sigma.detach().cpu().numpy().reshape(-1)
                    )

                wandb.log(log_dict, step=global_step)

                print(f"[learner {global_step}] critic_loss={metrics.get('train/critic_loss', -1):.4f}")

            # --------------------------------------------------------------
            # (D) Checkpoint
            # --------------------------------------------------------------
            if global_step > 0 and global_step % cfg.checkpoint_interval == 0:
                ckpt_path = model_save_dir / f"checkpoint_{global_step}.pt"
                save_training_checkpoint(
                    ckpt_path=ckpt_path,
                    agent=agent,
                    global_step=global_step,
                    best_eval_success_rate=best_eval_success_rate,
                    training_cum_time=training_cum_time,
                    episode_count=episode_count,
                    actor_updates=actor_updates,
                    run_name=run_name,
                    cfg=cfg,
                )

            if global_step >= cfg.algo.total_timesteps and episode_queue.empty():
                stop_event.set()
                break
                
            global_step +=1

        print("[learner] finished")

# -----------------------------------------------------------------------------
# Main training loop -----------------------------------------------------------
# -----------------------------------------------------------------------------
def main(cfg: ResidualTD3DexmgConfig):
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    enc_type=cfg.agent.enc_type

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


    # Load dataset and get normalization functions early
    print("Loading dataset and setting up normalization...")
    dataset = LeRobotDataset(cfg.offline_data.name)  # todo
    print("stats keys:", list(dataset.meta.stats.keys()))




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
    base_policy = BasePolicy(main_host="127.0.0.1", main_port=8008, action_scaler= action_scaler, state_standardizer=state_standardizer) # todo


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
   
    cfg.eval_num_envs = min(cfg.eval_num_envs, cfg.eval_num_episodes)
    num_cpus_available = os.cpu_count() - 1 if os.cpu_count() is not None else 1
    cfg.eval_num_envs = min(num_cpus_available, cfg.eval_num_envs)

  
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
   

    lowdim_dim = 8  # 8+8
    # img_c, img_h, img_w =3, 224, 224
    img_c, img_h, img_w =3, 84, 84
    action_dim = 8
    lowdim_keys = ["observation.state", "observation.base_action"]

    # ---------------------------------------------------------------------
    # Language / task embedding -------------------------------------------
    # ---------------------------------------------------------------------
    lang_cfg: LanguageConfig = cfg.agent.language
    if lang_cfg.enabled:
        lang_embedder = LanguageEmbedder(
            emb_dim=lang_cfg.lang_emb_dim,
            device=device,
        )
        # task_emb = lang_embedder(cfg.task)
        # print(f"🗣  task='{cfg.task}' -> task_emb shape={tuple(task_emb.shape)}")
        dataset_task_prompts = _load_lerobot_task_prompts(dataset, cfg.offline_data.name)
        lowdim_keys = ["observation.state", "observation.base_action", lang_cfg.lang_emb_obs_key]
    else:
        lang_embedder = None
        dataset_task_prompts = {}
        lowdim_keys = ["observation.state", "observation.base_action"]

    # ---------------------------------------------------------------------
    # Networks ------------------------------------------------------------
    # ---------------------------------------------------------------------
    agent = QAgentLang(
        obs_shape=(img_c, img_h, img_w),
        prop_shape=(lowdim_dim,),
        action_dim=action_dim,
        rl_cameras=image_keys,
        cfg=cfg.agent,
        residual_actor=True,  # Enable residual actor mode
    ) 
    
    def _attach_task_emb(obs, task_emb):
        if not lang_cfg.enabled or task_emb is None or obs is None:
            return obs
        return _inject_task_emb(obs, task_emb, key=lang_cfg.lang_emb_obs_key)

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
        base_policy: BasePolicy | None = None,
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
            
            if done_flag:
                print(f"episode done at {step_id}")

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
                    task_index = int(sample["task_index"].item())
                    task_prompt = dataset_task_prompts[task_index]
                    base_action = base_policy.get_offline_action_base(raw_obs, task_prompt)  # base_action shape:(8,)
                base_action = torch.as_tensor(base_action, dtype=torch.float32)
                base_action_scaled = action_scaler.scale(base_action.cpu())
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
                raw_k = k.replace("observation.images.", "")
                curr_obs[k] = sample[raw_k].squeeze(0)
            curr_obs = process_image_batch(curr_obs, image_keys, enc_type, rb=True)  ###
            # Convert images to uint8 for memory-efficient storage
            to_uint8(curr_obs, image_keys)

            # Inject task embedding from the dataset's per-frame task_index.
            if lang_cfg.enabled:
                if lang_embedder is None:
                    raise RuntimeError("lang_embedder must be initialized when language is enabled")
                if "task_index" not in sample:
                    raise KeyError("Dataset sample is missing 'task_index'; cannot build task embedding")
                task_index = int(sample["task_index"].item())
                if task_index not in dataset_task_prompts:
                    raise KeyError(
                        f"task_index={task_index} was not found in tasks.jsonl "
                        f"(available: {sorted(dataset_task_prompts)})"
                    )
                task_prompt = dataset_task_prompts[task_index]
                curr_task_emb = lang_embedder(task_prompt)
                curr_obs[lang_cfg.lang_emb_obs_key] = curr_task_emb.detach().cpu()

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

    if len(online_rb) < cfg.algo.learning_starts and not loaded_online_from_cache and not getattr(cfg, "resume", False):
        print(f"Warm-up: filling online buffer with {cfg.algo.learning_starts - len(online_rb)} random steps…")
        # obs, _ = env.reset()
        # obs = base_policy.reset() # todo reset
        # task_one = "pick up the cube and place it into the bowl"
        # task_two = "pick up the cube from the bowl and place it outside the bowl"
        # task_prompt = task_one
        # task_emb = lang_embedder(task_prompt)
        obs, task_prompt = base_policy.reset()
        task_emb = lang_embedder(task_prompt)
        obs = _attach_task_emb(obs= obs,task_emb=task_emb)
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
            next_obs, reward, done, info = base_policy.step(residual_action=rand_actions ) # todo need to return  reward, terminated, truncated, info  normalize 
            # print(f"[warmup] after step: reward={reward}, done={done}")
            task_prompt = info["task_prompt"]
            task_emb = lang_embedder(task_prompt)
            next_obs = _attach_task_emb(next_obs, task_emb)
            reward_sum += reward.sum().item()
            episode_count += done.float().sum().item()
            # reward_sum += reward
            # episode_count += int(done)
            if done:
                if reward>0:
                    res = "✓"
                else:
                    res = "✗"
                print(f"task: {task_prompt} : {res}")
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
    
    global_step = 0
    best_eval_success_rate = 0.0
    training_cum_time = 0.0
    episode_count = 0
    actor_updates = 0
    resume_state = None

    if getattr(cfg, "resume", False) and getattr(cfg, "resume_checkpoint", None):
        print(f"Resuming training from checkpoint: {cfg.resume_checkpoint}")
        resume_state = load_training_checkpoint(
            ckpt_path=cfg.resume_checkpoint,
            agent=agent,
            device=device,
        )

        global_step = resume_state["global_step"]
        best_eval_success_rate = resume_state["best_eval_success_rate"]
        training_cum_time = resume_state["training_cum_time"]
        episode_count = resume_state["episode_count"]
        actor_updates = resume_state["actor_updates"]

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

    if resume_state is not None and resume_state.get("run_name") is not None:
        run_name = resume_state["run_name"]
    else:
        run_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}__{hp_str}__seed{cfg.seed}"

        if cfg.wandb.name is not None:
            run_name = f"{cfg.wandb.name}__{run_name}"

    wandb_run_id = cfg.wandb.continue_run_id
    resume_checkpoint_run = getattr(cfg.wandb, "resume_checkpoint_run", True)
    if wandb_run_id is None and resume_state is not None and resume_checkpoint_run:
        wandb_run_id = resume_state.get("wandb_run_id")
    elif wandb_run_id is None and resume_state is not None and not resume_checkpoint_run:
        print(
            "Loaded training state from checkpoint, but starting a new wandb run "
            "because wandb.resume_checkpoint_run=False. This is the right choice "
            "when the checkpoint step is behind the existing wandb history."
        )
        run_name = f"{run_name}__from_step{global_step}"

    _wandb_config = OmegaConf.to_container(cfg, resolve=True)
    # Remove notes from config if present
    assert isinstance(_wandb_config, dict)
    _wandb_config["wandb"].pop("notes", None)

    # Print a nice summary of the config
    print("Launching run with the following config:")
    pprint.pprint(_wandb_config)

    if resume_state is not None and wandb_run_id is None and resume_checkpoint_run:
        print(
            "Checkpoint does not contain wandb_run_id; pass "
            "wandb.continue_run_id=<run_id> to resume the original wandb run."
        )

    if resume_state is not None and wandb_run_id is not None:
        print(
            f"Continuing wandb run {wandb_run_id} from checkpoint step {global_step}. "
            "WandB history is append-only: if that run already has logs after this "
            "step, they will remain. Use wandb.resume_checkpoint_run=False to load "
            "the checkpoint into a fresh wandb run instead."
        )

    wandb.init(
        id=wandb_run_id,
        resume=None if wandb_run_id is None else "allow",
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
    # obs = base_policy.reset() # todo reset

    train_start_time = time.time()

    # Initialize timing utility
    training_timer = TrainingTimer()

    def _run_critic_warmup(
        agent, online_rb, offline_rb, cfg, device, training_timer, online_batch_size, offline_batch_size
    ):
        """Run critic-only updates for warmup phase."""
     
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

    

    


    def launch_async_training():
        episode_queue = queue.Queue()
        weights_queue = queue.Queue(maxsize=1)
        stop_event = threading.Event()
        
        
        collector_thread = threading.Thread(
            target=collector_loop,
            args=( cfg,
                    lang_cfg,
                    lang_embedder,
                    device,
                    base_policy,
                    image_keys,
                    enc_type,
                    episode_queue,
                    weights_queue,
                    stop_event,
                    training_timer,
                    run_name,
                    outputs_dir,
                    img_c,
                    img_h,
                    img_w,
                    lowdim_dim,
                    action_dim,
                    global_step,
                    episode_count,
                    best_eval_success_rate,
                    ),
            daemon=True,
        )

        collector_thread.start()

        learner_loop( cfg,
                    device,
                    agent,
                    online_rb,
                    offline_rb,
                    image_keys,
                    lowdim_keys,
                    enc_type,
                    episode_queue,
                    weights_queue,
                    stop_event,
                    model_save_dir,
                    run_name,
                    online_cache_dir,
                    online_cache_meta,
                    online_cache_hash,
                    global_step,
                    actor_updates,
                    episode_count,
                    best_eval_success_rate,
                    training_cum_time,)

        collector_thread.join()

    launch_async_training()
    
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
