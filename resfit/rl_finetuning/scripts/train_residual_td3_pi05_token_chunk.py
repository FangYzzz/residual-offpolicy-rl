# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Token-based, chunk-form residual TD3 training (RL-Token, distill + frozen readout).

Pipeline (mirrors the reference latent-distill-residual-chunk trainer, pi05 edition):

  1. Distill: stream offline demos, query the token PI0 server per chunk-start frame
     to get the VLA prefix tokens z_1:M, and PRETRAIN the RL-token readout
     (autoregressive reconstruction).  Then FREEZE it and cache it (config-hashed) so
     later runs skip pretraining.
  2. Offline buffer: assemble chunk macro-transitions from the demos; the frozen
     readout compresses each chunk-start's tokens to z_rl, which is stored as
     ``observation.rl_token`` (the raw token sequence is dropped).
  3. Online buffer + RL: the collector rolls out chunks; the learner mixes
     offline+online batches (offline_fraction>0) and trains the chunk actor/critic on
     the frozen z_rl.  Chunk TD uses ``gamma = γ^C`` and the chunk-accumulated reward,
     both stored directly per transition (no MultiStepTransform).

Only the compact z_rl is ever stored in the buffers — never the raw embeddings.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import copy
import hashlib
import json
import queue
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from omegaconf import OmegaConf
from tensordict import TensorDict
from torch.utils.data import DataLoader
from torchrl.data import LazyTensorStorage, TensorDictPrioritizedReplayBuffer
from tqdm import tqdm

from resfit.rl_finetuning.config.residual_td3_token import ResidualTD3TokenChunkConfig  # noqa: F401 (hydra registration)
from resfit.rl_finetuning.off_policy.common_utils import utils
from resfit.rl_finetuning.off_policy.rl.q_agent_token_chunk import QAgentTokenChunk
from resfit.rl_finetuning.utils.evaluate_franka_token_chunk import run_franka_token_evaluation
from resfit.rl_finetuning.utils.normalization import ActionScaler, StateStandardizer

from token_residual_robot import TokenBasePolicy

_CACHE_ROOT = Path(os.environ.get("CACHE_DIR", ".")).expanduser().resolve()
DISTILL_CACHE_DIR = _CACHE_ROOT / "distill_cache"
OFFLINE_CACHE_DIR = _CACHE_ROOT / "offline_token_cache"


# -----------------------------------------------------------------------------
# Distill checkpoint cache (RL-token readout)
# -----------------------------------------------------------------------------
def _distill_cache_path(cfg, token_dim: int) -> Path:
    r = cfg.token
    meta = {
        "dataset": cfg.offline_data.name,
        "num_episodes": cfg.offline_data.num_episodes,
        "token_dim": token_dim,
        "rl_token_dim": r.rl_token_dim,
        "d_model": r.readout_d_model,
        "layers": r.readout_layers,
        "heads": r.readout_heads,
        "max_tokens": r.max_tokens,
        "pretrain_steps": r.distill_pretrain_steps,
        "max_frames": r.distill_pretrain_max_frames,
    }
    h = hashlib.sha1(json.dumps(meta, sort_keys=True).encode()).hexdigest()[:10]  # noqa: S324
    return DISTILL_CACHE_DIR / f"readout_{h}.pt"


def _save_distill(agent, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"readout_state_dict": agent.readout.state_dict(),
                "meta": {"z_dim": agent.z_dim, "token_dim": agent.token_dim}}, path)
    print(f"[distill] saved RL-token readout to {path}")


def _load_distill_into_agent(agent, path) -> bool:
    path = Path(path)
    if not path.exists():
        return False
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert int(ckpt["meta"]["token_dim"]) == agent.token_dim, "distill ckpt token_dim mismatch"
    assert int(ckpt["meta"]["z_dim"]) == agent.z_dim, "distill ckpt z_dim mismatch"
    agent.load_distill_state_dict(ckpt["readout_state_dict"], strict=True)
    print(f"[distill] loaded RL-token readout from {path}")
    return True


# -----------------------------------------------------------------------------
# Offline demo scan (one VLA query per chunk-start frame)
# -----------------------------------------------------------------------------
IMG_KEYS = ["exterior_image_1_left", "wrist_image_left", "exterior_image_2_left"]


def _raw_obs_short(sample, device):
    """Extract the short-keyed raw obs (images + eef + gripper) for get_offline_tokens."""
    want = IMG_KEYS + ["eef_position", "gripper_position"]
    out = {}
    for w in want:
        for k in sample:
            if k == w or k.endswith(w):
                out[w] = sample[k].squeeze(0).to(device)
                break
        assert w in out, f"offline sample missing a key ending with '{w}'"
    return out


def _scan_offline_demos(dataset, base_policy, action_scaler, state_standardizer, num_episodes, chunk_len, device):
    """One pass over the demos. Reads per-frame state/gt/done (cheap) and queries the
    VLA ONCE per chunk-start frame (stride = chunk_len) for its base chunk + tokens.

    Returns episodes: dict[ep -> list[frame]] where frame has
      {state (S,), gt (A,) scaled, done bool} always, and for chunk-start frames also
      {base_chunk (C*A,) scaled, tokens (M, emb) fp16 cpu}.
    Plus token_seqs: list of chunk-start token tensors (for readout pretrain).
    """
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    episodes: dict[int, list[dict]] = defaultdict(list)
    token_seqs: list[torch.Tensor] = []
    ep_frame_idx: dict[int, int] = defaultdict(int)
    print("[distill] scanning offline demos (VLA query per chunk-start frame)...")
    for sample in tqdm(loader, desc="scan demos"):
        ep = int(sample["episode_index"].item())
        if num_episodes is not None and ep >= num_episodes:
            break
        t = ep_frame_idx[ep]
        ep_frame_idx[ep] += 1

        if "next.done" in sample:
            done_flag = bool(sample["next.done"].item())
        elif "done" in sample:
            done_flag = bool(sample["done"].item())
        else:
            done_flag = False

        gt = action_scaler.scale(sample["eef_actions"].float().squeeze(0).to(device)).cpu()
        gp = sample["gripper_position"].float().squeeze(0)
        if gp.dim() == 0:
            gp = gp.reshape(1)
        state_raw = torch.cat([sample["eef_position"].float().squeeze(0), gp], dim=-1).to(device)
        state = state_standardizer.standardize(state_raw).cpu()

        frame = {"state": state, "gt": gt, "done": done_flag}
        if t % chunk_len == 0:  # chunk-start -> VLA query
            raw = _raw_obs_short(sample, device)
            base_chunk_unscaled, tokens = base_policy.get_offline_tokens(raw, base_policy.text)
            base_chunk_scaled = torch.stack(
                [action_scaler.scale(torch.as_tensor(a, dtype=torch.float32, device=device)) for a in base_chunk_unscaled],
                dim=0,
            ).reshape(-1).cpu()  # (C*A,)
            tok_t = torch.as_tensor(tokens, dtype=torch.float16)  # (M, emb) cpu
            frame["base_chunk"] = base_chunk_scaled
            frame["tokens"] = tok_t
            if len(token_seqs) < int(base_policy._distill_max_frames):
                token_seqs.append(tok_t)
        episodes[ep].append(frame)

    print(f"[distill] scanned {sum(len(v) for v in episodes.values())} frames across {len(episodes)} episodes; "
          f"{len(token_seqs)} token seqs for readout pretrain")
    return episodes, token_seqs


def _pretrain_and_freeze_readout(agent, token_seqs, cfg, device):
    if not token_seqs:
        print("[distill] no token seqs collected; skipping pretrain")
    else:
        Lmax = max(s.shape[0] for s in token_seqs)
        D = token_seqs[0].shape[1]
        N = len(token_seqs)
        seq_pool = torch.zeros(N, Lmax, D, dtype=torch.float16)
        mask = torch.ones(N, Lmax, dtype=torch.bool)  # True = pad
        for i, s in enumerate(token_seqs):
            L = s.shape[0]
            seq_pool[i, :L] = s
            mask[i, :L] = False
        bs = int(cfg.token.distill_pretrain_batch_size)
        print(f"[distill] pretraining readout: {cfg.token.distill_pretrain_steps} steps "
              f"(batch {bs}) on {N} seqs, Lmax={Lmax}, D={D}")
        for it in range(cfg.token.distill_pretrain_steps):
            idx = torch.randint(0, N, (min(bs, N),))
            m = agent.update_distill(seq_pool[idx].to(device).float(), mask[idx].to(device))
            if it % 200 == 0:
                print(f"  distill {it}/{cfg.token.distill_pretrain_steps} recon={m['rl_token/recon_loss']:.5f}")
    if cfg.token.freeze_after_pretrain:
        agent.freeze_distill()


# -----------------------------------------------------------------------------
# Offline buffer: chunk macro-transitions storing z_rl
# -----------------------------------------------------------------------------
def _cpu(t):
    return t.detach().to("cpu")


def _make_td(obs: dict, next_obs: dict, action, reward, done, gamma_chunk):
    done_t = torch.as_tensor(bool(done))
    return TensorDict(
        {
            "obs": TensorDict(obs, batch_size=[]),
            "action": _cpu(action).float(),
            "next": TensorDict(
                {
                    "obs": TensorDict(next_obs, batch_size=[]),
                    "done": done_t,
                    "reward": torch.tensor(float(reward), dtype=torch.float32),
                },
                batch_size=[],
            ),
            "gamma": torch.tensor(float(gamma_chunk), dtype=torch.float32),
            "nonterminal": ~done_t,
            "_priority": torch.tensor(10.0, dtype=torch.float32),
        },
        batch_size=[],
    )


def _populate_offline_buffer(episodes, agent, offline_rb, cfg, chunk_len, gamma, device):
    """Build chunk transitions (stride = chunk_len) from the scanned demos, storing the
    frozen-readout z_rl as observation.rl_token (raw tokens dropped)."""
    gamma_chunk = gamma ** chunk_len
    added = 0
    for _ep, frames in episodes.items():
        T = len(frames)
        starts = list(range(0, T - 1, chunk_len))
        for si in range(len(starts) - 1):
            s = starts[si]
            ns = starts[si + 1]
            if "base_chunk" not in frames[s] or "base_chunk" not in frames[ns]:
                continue
            # executed reward + terminal over [s, ns)
            R, term = 0.0, False
            for h in range(ns - s):
                r = float(frames[s + h]["done"])  # sparse: reward at terminal frame
                R += (gamma ** h) * r
                if frames[s + h]["done"]:
                    term = True
                    break
            # GT action chunk = demo actions [s, s+C) (pad with last).
            gt_rows = [frames[min(s + h, T - 1)]["gt"] for h in range(chunk_len)]
            action = torch.stack(gt_rows, dim=0).reshape(-1)  # (C*A,)

            with torch.no_grad():
                z_s = agent.encode_rl_token(
                    {"observation.vla_tokens": frames[s]["tokens"].to(device)}
                ).reshape(-1).cpu()
                z_ns = agent.encode_rl_token(
                    {"observation.vla_tokens": frames[ns]["tokens"].to(device)}
                ).reshape(-1).cpu()

            obs = {
                "observation.state": frames[s]["state"],
                "observation.base_action": frames[s]["base_chunk"],
                "observation.rl_token": z_s,
            }
            next_obs = {
                "observation.state": frames[ns]["state"],
                "observation.base_action": frames[ns]["base_chunk"],
                "observation.rl_token": z_ns,
            }
            offline_rb.add(_make_td(obs, next_obs, action, R, term, gamma_chunk))
            added += 1
    print(f"[offline] added {added} chunk transitions (size={len(offline_rb)})")
    return added


# -----------------------------------------------------------------------------
# Online transition store (compute z_rl with the frozen readout at add time)
# -----------------------------------------------------------------------------
def _pick_online_obs(o: dict, agent, device) -> dict:
    state = torch.as_tensor(o["observation.state"], device=device).float()
    if state.dim() == 2 and state.size(0) == 1:
        state = state.squeeze(0)
    base = torch.as_tensor(o["observation.base_action"], device=device).float()
    if base.dim() == 2 and base.size(0) == 1:
        base = base.squeeze(0)
    if "observation.rl_token" in o:
        z = torch.as_tensor(o["observation.rl_token"], device=device).float()
    else:
        z = agent.encode_rl_token(o).float()
    if z.dim() == 2 and z.size(0) == 1:
        z = z.squeeze(0)
    return {
        "observation.state": _cpu(state),
        "observation.base_action": _cpu(base),
        "observation.rl_token": _cpu(z),
    }


def _add_online_chunk(*, obs, next_obs, action, reward, done, gamma_chunk, agent, online_rb, device):
    action = torch.as_tensor(action, device=device).float().reshape(-1)
    online_rb.add(
        _make_td(
            _pick_online_obs(obs, agent, device),
            _pick_online_obs(next_obs, agent, device),
            action,
            float(reward.item() if torch.is_tensor(reward) else reward),
            bool(done.item() if torch.is_tensor(done) else done),
            gamma_chunk,
        )
    )


# -----------------------------------------------------------------------------
# Collector
# -----------------------------------------------------------------------------
def collector_loop(cfg, base_policy, agent_kwargs, initial_obs, episode_queue, weights_queue,
                   stop_event, distill_ckpt_path, run_name, output_dir, initial_global_step=0):
    global_step = initial_global_step
    episode_idx = 0
    agent = QAgentTokenChunk(**agent_kwargs)
    _load_distill_into_agent(agent, distill_ckpt_path)
    agent.freeze_distill()

    def _load_latest():
        latest = None
        while True:
            try:
                latest = weights_queue.get_nowait()
            except queue.Empty:
                break
        if latest is not None:
            agent.actor.load_state_dict(latest["actor"])
            agent.critic.load_state_dict(latest["critic"])

    obs = initial_obs
    while (global_step <= cfg.algo.total_timesteps) and (not stop_event.is_set()):
        _load_latest()
        steps = []
        episode_done = False
        while (not episode_done and global_step <= cfg.algo.total_timesteps
               and not stop_event.is_set() and len(steps) < cfg.send_transitions_len):
            with torch.no_grad(), utils.eval_mode(agent):
                stddev = utils.schedule(cfg.algo.stddev_schedule, global_step)
                residual = agent.act(copy.copy(obs), eval_mode=False, stddev=stddev, cpu=False)
            next_obs, reward, done, info = base_policy.step_chunk(residual_chunk=residual)
            if done.any():
                episode_done = True
                wandb.log({"training/reward": float(reward.item())}, step=global_step)
            steps.append({"obs": obs, "next_obs": next_obs, "action": info["scaled_action"],
                          "reward": reward, "done": done})
            obs = next_obs
            global_step += 1

        episode_queue.put({"global_step": global_step, "steps": steps})
        print(f"[collector] pushed chunk-episode {episode_idx}, len={len(steps)}, step={global_step}")
        episode_idx += 1

        if global_step % cfg.eval_interval_every_steps == 0 and (cfg.eval_first or global_step > 0):
            _load_latest()
            m = run_franka_token_evaluation(
                env=base_policy, agent=agent, num_episodes=cfg.eval_num_episodes,
                device=agent.cfg.device, global_step=global_step,
                save_video=cfg.save_video, save_q_plots=cfg.save_video,
                run_name=run_name, output_dir=output_dir,
            )
            print(f"🎉 task one SR: {m['eval/success_rate_task_one']}  task two SR: {m['eval/success_rate_task_two']}")
            obs, _ = base_policy.reset()

    stop_event.set()
    print("[collector] finished")


# -----------------------------------------------------------------------------
# Learner
# -----------------------------------------------------------------------------
def learner_loop(cfg, device, agent, online_rb, offline_rb, episode_queue, weights_queue,
                 stop_event, model_save_dir, gamma_chunk, initial_global_step=0):
    global_step = initial_global_step
    actor_updates = 0
    metrics = {}
    train_start = time.time()

    online_bs = int(cfg.algo.batch_size * (1 - cfg.algo.offline_fraction))
    offline_bs = cfg.algo.batch_size - online_bs

    def _publish():
        payload = {
            "actor": {k: v.detach().cpu() for k, v in agent.actor.state_dict().items()},
            "critic": {k: v.detach().cpu() for k, v in agent.critic.state_dict().items()},
        }
        try:
            while True:
                weights_queue.get_nowait()
        except queue.Empty:
            pass
        weights_queue.put(payload)

    _publish()

    while (not stop_event.is_set()) or (not episode_queue.empty()):
        try:
            ep = episode_queue.get_nowait()
        except queue.Empty:
            time.sleep(0.01)
            continue
        global_step = max(global_step, ep["global_step"])
        for st in ep["steps"]:
            _add_online_chunk(obs=st["obs"], next_obs=st["next_obs"], action=st["action"],
                              reward=st["reward"], done=st["done"], gamma_chunk=gamma_chunk,
                              agent=agent, online_rb=online_rb, device=device)

        if len(online_rb) < max(cfg.algo.learning_starts, online_bs):
            continue

        actor_cadence = max(1, cfg.algo.num_updates_per_iteration // cfg.algo.actor_updates_per_iteration)
        for i in range(cfg.algo.num_updates_per_iteration):
            online_batch = online_rb.sample(online_bs).to(device, non_blocking=True)
            if cfg.algo.offline_fraction > 0.0 and len(offline_rb) > 0:
                offline_batch = offline_rb.sample(offline_bs).to(device, non_blocking=True)
                batch = torch.cat([online_batch, offline_batch], dim=0)
            else:
                batch = online_batch
            update_actor = (i + 1) % actor_cadence == 0
            stddev = utils.schedule(cfg.algo.stddev_schedule, global_step)
            metrics = agent.update(batch, stddev, update_actor)
            if cfg.algo.sampling_strategy == "prioritized_replay" and "_td_errors" in metrics:
                batch["_priority"] = metrics["_td_errors"]
                online_rb.update_tensordict_priority(batch[:online_bs])
                if cfg.algo.offline_fraction > 0.0 and len(offline_rb) > 0:
                    offline_rb.update_tensordict_priority(batch[online_bs:])
            if update_actor:
                actor_updates += 1

        _publish()

        if global_step % cfg.log_freq == 0 and metrics:
            log = {
                "training/global_step": global_step,
                "training/actor_updates": actor_updates,
                "buffer/online_size": len(online_rb),
                "buffer/offline_size": len(offline_rb),
                "training/SPS": global_step / max(1e-6, time.time() - train_start),
            }
            log.update({k: v for k, v in metrics.items() if not k.startswith("_")})
            if "_actions" in metrics:
                log["train/residual_l2_magnitude"] = torch.mean(metrics["_actions"] ** 2).item()
                log["histograms/residual_actions"] = wandb.Histogram(metrics["_actions"].numpy().reshape(-1))
            if "_target_q" in metrics:
                log["histograms/critic_qt"] = wandb.Histogram(metrics["_target_q"].numpy().reshape(-1))
            wandb.log(log, step=global_step)
            print(f"[learner {global_step}] critic_loss={metrics.get('train/critic_loss', -1):.4f}")

        if cfg.checkpoint_interval > 0 and global_step % cfg.checkpoint_interval == 0:
            ckpt = model_save_dir / f"checkpoint_{global_step}.pt"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"global_step": global_step, "agent": agent.state_dict(),
                        "cfg": OmegaConf.to_container(cfg, resolve=True)}, ckpt)
            print(f"[learner] saved {ckpt}")

    print("[learner] finished")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def _make_buffer(cfg, max_size, batch_size):
    return TensorDictPrioritizedReplayBuffer(
        storage=LazyTensorStorage(max_size=max_size, device="cpu"),
        alpha=cfg.algo.priority_alpha if cfg.algo.sampling_strategy == "prioritized_replay" else 0.0,
        beta=cfg.algo.priority_beta if cfg.algo.sampling_strategy == "prioritized_replay" else 0.0,
        eps=1e-6,
        priority_key="_priority",
        batch_size=max(1, batch_size),
    )


def main(cfg: ResidualTD3TokenChunkConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    C = cfg.token.chunk_len
    A = cfg.token.per_step_action_dim
    chunk_action_dim = C * A
    prop_dim = 8

    print("Loading dataset for normalization stats...")
    dataset = LeRobotDataset(cfg.offline_data.name)
    action_scaler = ActionScaler.from_dataset_stats(
        action_stats=dataset.meta.stats["eef_actions"],
        action_scale=cfg.agent.actor.action_scale,
        min_range_per_dim=cfg.offline_data.min_action_range,
        device=device,
    )

    def concat_state_stats(a, b):
        out = {}
        for k in ["mean", "std", "min", "max"]:
            if k in a and k in b:
                out[k] = torch.cat(
                    [torch.as_tensor(a[k], dtype=torch.float32), torch.as_tensor(b[k], dtype=torch.float32)], dim=-1
                )
        return out

    state_standardizer = StateStandardizer.from_dataset_stats(
        state_stats=concat_state_stats(dataset.meta.stats["eef_position"], dataset.meta.stats["gripper_position"]),
        min_std=cfg.offline_data.min_state_std,
        device=device,
    )

    base_policy = TokenBasePolicy(
        main_host="127.0.0.1", main_port=8008,
        action_scaler=action_scaler, state_standardizer=state_standardizer,
        chunk_len=C, gamma=cfg.algo.gamma,
    )
    base_policy._distill_max_frames = cfg.token.distill_pretrain_max_frames

    # Detect token dims from the first VLA response.
    initial_obs, _ = base_policy.reset()
    tokens0 = initial_obs["observation.vla_tokens"]
    max_tokens, token_dim = int(tokens0.shape[-2]), int(tokens0.shape[-1])
    cfg.token.token_dim, cfg.token.max_tokens = token_dim, max(max_tokens, cfg.token.max_tokens)
    print(f"Detected VLA tokens: M={max_tokens}, emb={token_dim}. chunk_action_dim={chunk_action_dim}")

    agent_kwargs = dict(
        token_dim=token_dim, max_tokens=cfg.token.max_tokens, prop_dim=prop_dim, action_dim=chunk_action_dim,
        cfg=cfg.agent, rl_token_dim=cfg.token.rl_token_dim, readout_d_model=cfg.token.readout_d_model,
        readout_layers=cfg.token.readout_layers, readout_heads=cfg.token.readout_heads,
        readout_dropout=cfg.token.readout_dropout, distill_lr=cfg.token.distill_lr,
        distill_grad_clip_norm=cfg.token.distill_grad_clip_norm, ref_action_dropout=cfg.token.ref_action_dropout,
    )
    agent = QAgentTokenChunk(**agent_kwargs)

    # ---- Distill: load cached readout, else pretrain on demos + freeze + cache ----
    distill_ckpt_path = Path(cfg.token.distill_ckpt) if cfg.token.distill_ckpt else _distill_cache_path(cfg, token_dim)
    gamma_chunk = cfg.algo.gamma ** C

    online_rb = _make_buffer(cfg, cfg.algo.buffer_size, int(cfg.algo.batch_size * (1 - cfg.algo.offline_fraction)))
    offline_rb = _make_buffer(cfg, max(cfg.algo.buffer_size, 1), max(int(cfg.algo.batch_size * cfg.algo.offline_fraction), 1))

    if _load_distill_into_agent(agent, distill_ckpt_path):
        print(f"Found distill ckpt {distill_ckpt_path}; skipping pretrain. Rebuilding offline buffer...")
        agent.freeze_distill()
        episodes, _ = _scan_offline_demos(dataset, base_policy, action_scaler, state_standardizer,
                                          cfg.offline_data.num_episodes, C, device)
    else:
        episodes, token_seqs = _scan_offline_demos(dataset, base_policy, action_scaler, state_standardizer,
                                                   cfg.offline_data.num_episodes, C, device)
        _pretrain_and_freeze_readout(agent, token_seqs, cfg, device)
        _save_distill(agent, distill_ckpt_path)

    if cfg.algo.offline_fraction > 0.0:
        _populate_offline_buffer(episodes, agent, offline_rb, cfg, C, cfg.algo.gamma, device)
    del episodes

    run_name = f"token_chunk_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_C{C}"
    wandb.init(project=cfg.wandb.project, name=run_name,
               config=OmegaConf.to_container(cfg, resolve=True),
               mode=cfg.wandb.mode if not cfg.debug else "disabled")
    run_dir = _CACHE_ROOT / f"run_{run_name}"
    model_save_dir, outputs_dir = run_dir / "models", run_dir / "outputs"

    episode_queue: queue.Queue = queue.Queue()
    weights_queue: queue.Queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()

    collector_thread = threading.Thread(
        target=collector_loop,
        args=(cfg, base_policy, agent_kwargs, initial_obs, episode_queue, weights_queue,
              stop_event, distill_ckpt_path, run_name, outputs_dir, 0),
        daemon=True,
    )
    collector_thread.start()
    learner_loop(cfg, device, agent, online_rb, offline_rb, episode_queue, weights_queue,
                 stop_event, model_save_dir, gamma_chunk, 0)
    collector_thread.join()
    print("Training finished.")


@hydra.main(version_base=None, config_name="residual_td3_token_chunk_config")
def hydra_entry(cfg: ResidualTD3TokenChunkConfig):
    main(OmegaConf.structured(cfg))


if __name__ == "__main__":
    hydra_entry()
