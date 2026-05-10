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
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.scripts.residual_client import ResidualClient


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


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# Main eval loop
# ----------------------------------------------------------------------
def run_franka_evaluation(
    *,
    env: ResidualClient,
    agent: QAgent,
    num_episodes: int = 20,
    device: torch.device | str = "cuda",
    global_step: int | None = None,
    save_video: bool = False,
    save_q_plots: bool = True,
    run_name: str | None = None,
    output_dir: str | Path | None = "outputs",
) -> tuple[dict[str, float], float]:
    device = torch.device(device)
    agent.eval()

    num_envs: int = env.num_envs if hasattr(env, "num_envs") else 1

    successes: list[bool] = []

    # 每 episode 的缓冲
    ep_combined_buffers: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
    ep_residual_buffers: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
    ep_q_preds: list[list[float]] = [[] for _ in range(num_envs)]

    # 已完成 episode 聚合
    all_combined_trajs: list[np.ndarray] = []
    all_residual_trajs: list[np.ndarray] = []
    all_q_trajectories: list[list[float]] = []
    all_episode_lengths: list[int] = []

    frame_buffer: list[list[np.ndarray]] | None = [[] for _ in range(num_envs)] if save_video else None

    done_episodes = 0
    obs = env.reset()

    progress_dots = ["."] * num_episodes
    print(f"Evaluating {num_episodes} episodes: {''.join(progress_dots)}", end="", flush=True)
    image_keys = [
        # "observation.images.head_view",
        # "observation.images.wrist_right_view",
        # "observation.images.wrist_left_view",
        "observation.images.wrist_image_left",
        "observation.images.exterior_image_1_left",
        "observation.images.exterior_image_2_left",
    ]

    while done_episodes < num_episodes:
        # -----------------------------------------------------------
        # 1. Policy + Q prediction
        # -----------------------------------------------------------
        with torch.no_grad():
            obs = process_image_batch_dim(obs, image_keys, out_size=84)
            actions = q_actions = agent.act(obs, eval_mode=True, stddev=0.0, cpu=False)

            # On-the-fly features for Q-value prediction
            obs_q = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in obs.items()}
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
        next_obs, reward, done, info = env.step(actions)
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
            print("rw:", reward.item())
            progress_dots[done_episodes] = "✓" if is_success else "✗"
            print(f"\rEvaluating {num_episodes} episodes: {''.join(progress_dots)}", end="", flush=True)

            successes.append(is_success)

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

            all_q_trajectories.append(ep_q_preds[0].copy())
            all_episode_lengths.append(len(ep_q_preds[0]))

            ep_combined_buffers[0] = []
            ep_residual_buffers[0] = []
            ep_q_preds[0] = []

            done_episodes += 1

        obs = next_obs

    print("Done")

    # ------------------------------------------------------------------
    # 5. Aggregate metrics + Q stats
    # ------------------------------------------------------------------
    success_rate: float = float(np.mean(successes)) if successes else 0.0

    flat_q = (
        np.concatenate([np.asarray(t, dtype=np.float32) for t in all_q_trajectories if len(t) > 0])
        if any(len(t) > 0 for t in all_q_trajectories)
        else np.zeros((0,), dtype=np.float32)
    )
    succ_q = [np.mean(t) for t, ok in zip(all_q_trajectories, successes) if ok and len(t) > 0]
    fail_q = [np.mean(t) for t, ok in zip(all_q_trajectories, successes) if (not ok) and len(t) > 0]

    metrics: dict[str, float] = {
        "eval/success_rate": success_rate,
        "eval/q_mean": float(flat_q.mean()) if flat_q.size else 0.0,
        "eval/q_std": float(flat_q.std()) if flat_q.size else 0.0,
        "eval/q_min": float(flat_q.min()) if flat_q.size else 0.0,
        "eval/q_max": float(flat_q.max()) if flat_q.size else 0.0,
        "eval/q_mean_success": float(np.mean(succ_q)) if succ_q else 0.0,
        "eval/q_mean_failure": float(np.mean(fail_q)) if fail_q else 0.0,
    }

    # ------------------------------------------------------------------
    # 6. Plots -> wandb
    # ------------------------------------------------------------------
    fig_combined = _plot_xyz_trajectories(
        all_combined_trajs, successes,
        title_prefix="Combined action", global_step=global_step,
    )
    fig_residual = _plot_xyz_trajectories(
        all_residual_trajs, successes,
        title_prefix="Residual action", global_step=global_step,
    )
    fig_q = _plot_q_trajectories(all_q_trajectories, successes, global_step=global_step) \
        if save_q_plots else None

    if wandb.run is not None:
        log_dict = {
            **metrics,
            "eval/xyz_trajectories_combined": wandb.Image(fig_combined),
            "eval/xyz_trajectories_residual": wandb.Image(fig_residual),
        }
        if fig_q is not None:
            log_dict["eval/q_trajectories"] = wandb.Image(fig_q)
        wandb.log(log_dict, step=global_step)

    plt.close(fig_combined)
    plt.close(fig_residual)
    if fig_q is not None:
        plt.close(fig_q)

    agent.train(True)
    return metrics