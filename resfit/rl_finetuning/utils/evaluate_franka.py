# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

from pathlib import Path

import imageio
import matplotlib
matplotlib.use("Agg")   # 必须在 import pyplot 之前
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

import wandb
# from resfit.dexmg.environments.dexmg import VectorizedEnvWrapper
from resfit.rl_finetuning.off_policy.rl.q_agent_lang import QAgentLang
from resfit.rl_finetuning.scripts.gpt_residual_robot import BasePolicy
from loguru import logger
import sys

def process_image_batch_dim(obs_dict, image_keys, out_size=84):
    """
    将 obs_dict 里的多个图像键统一处理成 [3, out_size, out_size]
    输入每张图预期是 [H, W, C] 或 [1, H, W, C]
    输出每张图是 float32, [3, out_size, out_size], range [0, 1]
    """
    imgs = []
    for k in image_keys:
        x = obs_dict[k]
        if x.ndim == 4 and x.shape[0] == 1:
            x = x.squeeze(0)
        if x.ndim != 3:
            raise ValueError(f"{k} expected 3 dims [H,W,C], got shape={x.shape}")

        if x.shape[-1] == 3:
            x = x.permute(2, 0, 1)
        elif x.shape[0] == 3:
            pass
        else:
            raise ValueError(f"{k} is neither HWC nor CHW, got shape={x.shape}")

        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        else:
            x = x.float()
        imgs.append(x)

    imgs = torch.stack(imgs, dim=0)
    imgs = F.interpolate(imgs, size=(out_size, out_size), mode="bilinear", align_corners=False)

    for i, k in enumerate(image_keys):
        obs_dict[k] = imgs[i].contiguous().unsqueeze(0)
    return obs_dict


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _to_xyz_steps(arr) -> np.ndarray:
    """把 action 张量/数组归一成 [T, 3] 的 xyz 序列。"""
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)

    if arr.ndim == 3:        # [B, T, A] -> batch 0
        return arr[0, :, :3]
    elif arr.ndim == 2:      # [T, A]
        return arr[:, :3]
    elif arr.ndim == 1:      # [A]
        return arr[None, :3]
    else:
        return arr.reshape(-1)[None, :3]

def setup_logger(log_dir: str | Path):
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "log.txt"

        logger.remove()
        logger.add(str(log_path), format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO")
        logger.add(sys.stdout, colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------


def _plot_residual_per_dim(
    trajectories: list[np.ndarray],
    successes: list[bool],
    title_prefix: str = "Residual action",
    global_step: int | None = None,
):
    """
    Residual action 可视化：
      - x 轴: episode step
      - y 轴: residual action 数值
      - 三个子图分别对应 x / y / z 维度
      - 成功 episode 绿色，失败红色
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    dim_names = ["x", "y", "z"]

    drew_success_label = False
    drew_fail_label = False
    for traj, ok in zip(trajectories, successes):
        if traj.shape[0] == 0:
            continue
        steps = np.arange(traj.shape[0])
        color = "tab:green" if ok else "tab:red"

        for d in range(3):
            label = None
            if ok and not drew_success_label and d == 0:
                label = "success"
            elif (not ok) and not drew_fail_label and d == 0:
                label = "fail"
            axes[d].plot(
                steps, traj[:, d],
                color=color, alpha=0.6, linewidth=1.0, label=label,
            )
        if ok:
            drew_success_label = True
        else:
            drew_fail_label = True

    title_suffix = f" (step={global_step})" if global_step is not None else ""
    for d, name in enumerate(dim_names):
        axes[d].set_ylabel(f"residual {name}")
        axes[d].grid(True, alpha=0.3)
        axes[d].axhline(0.0, color="k", linewidth=0.5, alpha=0.4)
    axes[0].set_title(f"{title_prefix} per-dim over steps{title_suffix}")
    axes[-1].set_xlabel("episode step")
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(loc="best", fontsize=8)

    fig.tight_layout()
    return fig


def _plot_xyz_trajectories(
    trajectories: list[np.ndarray],
    successes: list[bool],
    title_prefix: str = "Combined action",
    global_step: int | None = None,
):
    """3D 轨迹 + 三视图。成功绿色，失败红色。"""
    fig = plt.figure(figsize=(14, 10))
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax_xy = fig.add_subplot(2, 2, 2)
    ax_xz = fig.add_subplot(2, 2, 3)
    ax_yz = fig.add_subplot(2, 2, 4)

    for i, (traj, ok) in enumerate(zip(trajectories, successes)):
        if traj.shape[0] == 0:
            continue
        color = "tab:green" if ok else "tab:red"
        alpha = 0.75
        label = None
        if ok and not any(s for s in successes[:i]):
            label = "success"
        elif (not ok) and not any((not s) for s in successes[:i]):
            label = "fail"

        x, y, z = traj[:, 0], traj[:, 1], traj[:, 2]
        ax3d.plot(x, y, z, color=color, alpha=alpha, linewidth=1.2, label=label)
        ax3d.scatter(x[0], y[0], z[0], color=color, marker="o", s=20)
        ax3d.scatter(x[-1], y[-1], z[-1], color=color, marker="x", s=30)

        ax_xy.plot(x, y, color=color, alpha=alpha, linewidth=1.0)
        ax_xz.plot(x, z, color=color, alpha=alpha, linewidth=1.0)
        ax_yz.plot(y, z, color=color, alpha=alpha, linewidth=1.0)

    ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("z")
    title_suffix = f" (step={global_step})" if global_step is not None else ""
    ax3d.set_title(f"{title_prefix} xyz trajectories{title_suffix}")
    if ax3d.get_legend_handles_labels()[0]:
        ax3d.legend(loc="best", fontsize=8)

    ax_xy.set_xlabel("x"); ax_xy.set_ylabel("y"); ax_xy.set_title("xy projection"); ax_xy.grid(True, alpha=0.3)
    ax_xz.set_xlabel("x"); ax_xz.set_ylabel("z"); ax_xz.set_title("xz projection"); ax_xz.grid(True, alpha=0.3)
    ax_yz.set_xlabel("y"); ax_yz.set_ylabel("z"); ax_yz.set_title("yz projection"); ax_yz.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def _plot_q_trajectories(
    trajectories: list[list[float]],
    successes: list[bool],
    global_step: int | None = None,
):
    """
    Q-value 可视化：
      - 上图：每个 episode 一条 Q(t) 折线，成功绿、失败红
      - 下图：在 episode 25/50/75/100 % 处的 Q 分布 boxplot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    drew_success_label = False
    drew_fail_label = False
    for traj, ok in zip(trajectories, successes):
        if not traj:
            continue
        steps = list(range(len(traj)))
        if ok:
            label = None if drew_success_label else "Success"
            drew_success_label = True
            ax1.plot(steps, traj, "g-", alpha=0.6, linewidth=1, label=label)
        else:
            label = None if drew_fail_label else "Failure"
            drew_fail_label = True
            ax1.plot(steps, traj, "r-", alpha=0.6, linewidth=1, label=label)

    title_suffix = f" (step={global_step})" if global_step is not None else ""
    ax1.set_xlabel("Episode Step")
    ax1.set_ylabel("Q-Value")
    ax1.set_title(f"Q-Value Trajectories Over Time{title_suffix}")
    ax1.grid(True, alpha=0.3)
    if drew_success_label or drew_fail_label:
        ax1.legend()

    progress_points = [0.25, 0.5, 0.75, 1.0]
    bucket: dict[str, list[float]] = {f"{int(p * 100)}%": [] for p in progress_points}
    for traj in trajectories:
        if not traj:
            continue
        L = len(traj)
        for p in progress_points:
            idx = min(int(p * L), L - 1)
            bucket[f"{int(p * 100)}%"].append(traj[idx])

    box_data = [bucket[f"{int(p * 100)}%"] for p in progress_points]
    box_labels = [f"{int(p * 100)}%" for p in progress_points]
    ax2.boxplot(box_data, labels=box_labels)
    ax2.set_xlabel("Episode Progress")
    ax2.set_ylabel("Q-Value")
    ax2.set_title("Q-Value Distribution at Different Episode Progress Points")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig



def _inject_task_emb(
    obs: dict[str, torch.Tensor], task_emb: torch.Tensor, key: str = "observation.task_emb"
) -> dict[str, torch.Tensor]:
    """Set ``obs[key]`` to ``task_emb`` (broadcast to whatever batch dim the
    rest of obs has).  Mutates ``obs`` in place and returns it for chaining.
    """
    if key in obs:
        return obs
    state = obs.get("observation.state", None)
    if state is not None and state.dim() == 2:
        emb = task_emb.unsqueeze(0).expand(state.size(0), -1).contiguous()
    else:
        emb = task_emb
    obs[key] = emb.to(state.device if state is not None else task_emb.device)
    return obs

def _attach_task_emb(obs, lang_cfg, task_emb):
            if not lang_cfg.enabled or task_emb is None or obs is None:
                return obs
            return _inject_task_emb(obs, task_emb, key=lang_cfg.lang_emb_obs_key)

# ----------------------------------------------------------------------
# Main eval loop
# ----------------------------------------------------------------------
def run_franka_evaluation(
    *,
    env: BasePolicy,
    agent: QAgentLang,
    eval_num_episode: int = 20,
    device: torch.device | str = "cuda",
    global_step: int | None = None,
    save_video: bool = False,
    save_q_plots: bool = True,
    run_name: str | None = None,
    output_dir: str | Path | None = "outputs",
    lang_cfg,
    lang_embedder,
    chunk_len: int = 1,
) -> tuple[dict[str, float], float]:
    device = torch.device(device)
    agent.eval()
    num_envs: int = env.num_envs if hasattr(env, "num_envs") else 1
    eval_step = global_step or 0
    cycle_dir = env.task_reward_generator.set_output_cycle(eval_step)
    setup_logger(cycle_dir)
    logger.info(f"---------------- evaluation after {eval_step} steps ----------------")
    candidate_tasks = list(env.task_reward_generator.candidate_tasks)
    if not candidate_tasks:
        raise ValueError("env.task_reward_generator.candidate_tasks is empty")
    logger.info(f"Evaluating each task for {eval_num_episode} episodes:")
    for index, task in enumerate(candidate_tasks, start=1):
        logger.info(f"task {index}: {task}")

    successes_by_task: dict[str, list[bool]] = {
        task: [] for task in candidate_tasks
    }
    successes: list[bool] = []
    # 每 episode 的缓冲
    ep_combined_buffers: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
    ep_residual_buffers: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
    ep_q_preds: list[list[float]] = [[] for _ in range(num_envs)]

    # 已完成 episode 聚合
    all_combined_trajs: list[np.ndarray] = []
    all_residual_trajs: list[np.ndarray] = []
    all_q_trajectories_by_task: dict[str, list[list[float]]] = {
        task: [] for task in candidate_tasks
    }
    # all_episode_lengths: list[int] = []

    frame_buffer: list[list[np.ndarray]] | None = [[] for _ in range(num_envs)] if save_video else None

    done_episodes = 0
    total_episodes = eval_num_episode * len(candidate_tasks)
    task_index = 0
    task_prompt = candidate_tasks[task_index]
    logger.info(f"---------------- task {task_index + 1}: {task_prompt} ----------------")
    env.evaluation = True
    obs = env.reset(task_prompt, evaluate_previous=False)
    task_emb = lang_embedder(task_prompt)
    obs = _attach_task_emb(obs,lang_cfg, task_emb)

    progress_by_task = {
        task: ["."] * eval_num_episode for task in candidate_tasks
    }
    logger.info(
        f"Evaluating {eval_num_episode} episodes: {''.join(progress_by_task[task_prompt])}",
        end="",
        flush=True,
    )
    image_keys = [
        # "observation.images.head_view",
        # "observation.images.wrist_right_view",
        # "observation.images.wrist_left_view",
        "observation.images.wrist_image_left",
        # "observation.images.exterior_image_1_left",
        "observation.images.exterior_image_2_left",
    ]

    while done_episodes < total_episodes:
        # -----------------------------------------------------------
        # 1. Policy + Q prediction
        # -----------------------------------------------------------
        with torch.no_grad():
            obs = process_image_batch_dim(obs, image_keys, out_size=84)
            # obs = _attach_task_emb(obs,lang_cfg, task_emb)
            # Attach the base-action chunk (H*dim) so actor/critic see chunk-level base.
            obs["observation.base_action"] = env.current_base_chunk(chunk_len)
            actions = q_actions = agent.act(obs, eval_mode=True, stddev=0.0, cpu=False)

            obs_q = agent._augment_state(obs, detach_lang=True)
            # On-the-fly features for Q-value prediction
            obs_q = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in obs_q.items()}
            obs_q["feat"] = agent._encode(obs_q, augment=False)

            if getattr(agent, "residual_actor", False) and "observation.base_action" in obs:
                q_actions = torch.clamp(obs["observation.base_action"] + actions, -1.0, 1.0)

            q_pred = (
                agent.critic.q_value(obs_q["feat"], obs_q["observation.state"], q_actions)
                .detach().cpu().squeeze(-1)
            )
            q_pred = q_pred.reshape(-1)

        # -----------------------------------------------------------
        # 2. Env step
        # -----------------------------------------------------------
        # Open-loop execute the whole predicted residual chunk.
        next_obs, combined_chunk, reward, done, info = env.step_chunk(
            actions, task_prompt=task_prompt, evaluation=True
        )

        combined_action = info["combined_action"]
        residual_action = info["residual_action"]
        done_flags = done

        # -----------------------------------------------------------
        # 3. 采集 xyz + Q
        # -----------------------------------------------------------
        combined_xyz = _to_xyz_steps(combined_action)
        residual_xyz = _to_xyz_steps(residual_action)

        for env_idx in range(num_envs):
            ep_combined_buffers[env_idx].append(combined_xyz.copy())
            ep_residual_buffers[env_idx].append(residual_xyz.copy())
            q_val = q_pred[env_idx].item() if q_pred.numel() > env_idx else float(q_pred.mean().item())
            ep_q_preds[env_idx].append(q_val)

        # -----------------------------------------------------------
        # 4. Episode 结束
        # -----------------------------------------------------------
        if done_flags:
            is_success = bool(reward.item() > 0.9)           
            task_episode_index = len(successes_by_task[task_prompt])
            progress_by_task[task_prompt][task_episode_index] = "✓" if is_success else "✗"
            # logger.info(f"{reward}: {reward}")
            logger.info(f"{task_prompt}: {progress_by_task[task_prompt][task_episode_index]}")
            logger.info(
                f"Evaluating {eval_num_episode} episodes: "
                f"{''.join(progress_by_task[task_prompt])}",
                end="",
                flush=True,
            )
            successes_by_task[task_prompt].append(is_success)
            successes.append(is_success)
            if len(successes_by_task[task_prompt]) == eval_num_episode:
                task_successes = successes_by_task[task_prompt]
                success_count = sum(task_successes)
                success_rate = success_count / eval_num_episode
                logger.info(
                    f"task {task_index + 1} success rate: {success_rate:.2%} "
                    f"({success_count}/{eval_num_episode})"
                )
            combined_traj = (
                np.concatenate(ep_combined_buffers[0], axis=0)
                if ep_combined_buffers[0] else np.zeros((0, 3))
            )
            residual_traj = (
                np.concatenate(ep_residual_buffers[0], axis=0)
                if ep_residual_buffers[0] else np.zeros((0, 3))
            )
            all_combined_trajs.append(combined_traj)
            all_residual_trajs.append(residual_traj)

            all_q_trajectories_by_task[task_prompt].append(ep_q_preds[0].copy())
            # all_episode_lengths.append(len(ep_q_preds[0]))

            ep_combined_buffers[0] = []
            ep_residual_buffers[0] = []

            ep_q_preds[0] = []

            done_episodes += 1

            next_task_index = min(done_episodes // eval_num_episode, len(candidate_tasks) - 1)
            if done_episodes < total_episodes and next_task_index != task_index:
                task_index = next_task_index
                task_prompt = candidate_tasks[task_index]
                logger.info(f"---------------- task {task_index + 1}: {task_prompt} ----------------")
                next_obs = env.reset(task_prompt, evaluate_previous=False)
                logger.info(
                    f"Evaluating {eval_num_episode} episodes: "
                    f"{''.join(progress_by_task[task_prompt])}",
                    end="",
                    flush=True,
                )
    
            task_emb = lang_embedder(task_prompt)
        next_obs = _attach_task_emb(next_obs,lang_cfg, task_emb)
        obs = next_obs

    logger.info("Done")
    env.evaluation = False

    # ------------------------------------------------------------------
    # 5. Aggregate metrics + Q stats
    # ------------------------------------------------------------------
    metrics: dict[str, float] = {}
    for index, task in enumerate(candidate_tasks):
        task_successes = successes_by_task[task]
        task_q_trajectories = all_q_trajectories_by_task[task]
        nonempty_q = [
            np.asarray(trajectory, dtype=np.float32)
            for trajectory in task_q_trajectories
            if trajectory
        ]
        flat_q = np.concatenate(nonempty_q) if nonempty_q else np.zeros((0,), dtype=np.float32)
        metric_prefix = f"eval/task_{index + 1}"
        metrics[f"{metric_prefix}/success_rate"] = (
            float(np.mean(task_successes)) if task_successes else 0.0
        )
        metrics[f"{metric_prefix}/q_mean"] = (
            float(flat_q.mean()) if flat_q.size else 0.0
        )

    # ------------------------------------------------------------------
    # 6. Plots -> wandb
    # ------------------------------------------------------------------
    fig_combined = _plot_xyz_trajectories(
        all_combined_trajs, successes,
        title_prefix="Combined action", global_step=global_step,
    )
    fig_residual = _plot_residual_per_dim(
        all_residual_trajs, successes,
        title_prefix="Residual action", global_step=global_step,
    )
    q_figures = {
        task: _plot_q_trajectories(
            all_q_trajectories_by_task[task],
            successes_by_task[task],
            global_step=global_step,
        )
        for task in candidate_tasks
    } if save_q_plots else {}

    if wandb.run is not None:
        log_dict = {
            **metrics,
            "eval/xyz_trajectories_combined": wandb.Image(fig_combined),
            "eval/xyz_trajectories_residual": wandb.Image(fig_residual),
        }
        for index, task in enumerate(candidate_tasks):
            if task in q_figures:
                log_dict[f"eval/task_{index + 1}/q_trajectories"] = wandb.Image(q_figures[task])
        wandb.log(log_dict, step=global_step)

    plt.close(fig_combined)
    plt.close(fig_residual)
    for figure in q_figures.values():
        plt.close(figure)

    logger.info("--------------------------------------------------------------------------------------")

    agent.train(True)
    return metrics
