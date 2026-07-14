# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Chunk-form evaluation for the token-based residual agent (RL-Token method).

Mirrors ``evaluate_franka.run_franka_evaluation`` but for
:class:`QAgentTokenChunk` + :class:`TokenBasePolicy`:

* no image processing / no language embedding (the prompt is inside the VLA tokens);
* the policy acts over a whole action chunk per call (``env.step_chunk``), so each
  env interaction contributes one Q value and ``C`` xyz points to the trajectory.

The two-task structure (task one for the first ``num_episodes``, then switch to task
two) and the wandb plots (combined-xyz / residual-per-dim / Q trajectories) are kept
identical to the per-step evaluator, reusing its plotting helpers.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from loguru import logger

from resfit.rl_finetuning.off_policy.rl.q_agent_token_chunk import QAgentTokenChunk
from resfit.rl_finetuning.scripts.token_residual_robot import TokenBasePolicy
from resfit.rl_finetuning.utils.evaluate_franka import (
    _plot_q_trajectories,
    _plot_residual_per_dim,
    _plot_xyz_trajectories,
    _to_xyz_steps,
    setup_logger,
)


@torch.no_grad()
def _act_and_q(agent: QAgentTokenChunk, obs: dict[str, torch.Tensor]):
    """Return (residual_chunk [1, C*A], combined_chunk [1, C*A], q [1])."""
    obs_b, _ = QAgentTokenChunk._ensure_batched(obs)
    z_rl = agent._encode(obs_b)
    actor_obs = {
        "feat": z_rl,
        "observation.state": obs_b["observation.state"],
        "observation.base_action": obs_b["observation.base_action"],
    }
    residual = agent.actor.forward(actor_obs, 0.0).mean  # deterministic
    combined = torch.clamp(obs_b["observation.base_action"] + residual, -1.0, 1.0)
    q = agent.critic.q_value(z_rl, obs_b["observation.state"], combined).squeeze(-1).reshape(-1)
    return residual, combined, q


def run_franka_token_evaluation(
    *,
    env: TokenBasePolicy,
    agent: QAgentTokenChunk,
    num_episodes: int = 20,
    device: torch.device | str = "cuda",
    global_step: int | None = None,
    save_video: bool = False,
    save_q_plots: bool = True,
    run_name: str | None = None,
    output_dir: str | Path | None = "outputs",
) -> dict[str, float]:
    device = torch.device(device)
    agent.eval()
    setup_logger()
    logger.info("------------------------------- token eval --------------------------------")

    successes_task_one: list[bool] = []
    successes_task_two: list[bool] = []
    successes: list[bool] = []
    switched = False

    ep_combined: list[np.ndarray] = []
    ep_residual: list[np.ndarray] = []
    ep_q_one: list[float] = []
    ep_q_two: list[float] = []

    all_combined_trajs: list[np.ndarray] = []
    all_residual_trajs: list[np.ndarray] = []
    all_q_one: list[list[float]] = []
    all_q_two: list[list[float]] = []

    task_one = "put the cube into the bowl"
    task_two = "put the cube outside the bowl"

    obs, _ = env.reset()
    task_prompt = task_one

    done_episodes = 0
    progress_dots = ["."] * (num_episodes * 2)

    while done_episodes < num_episodes * 2:
        # 1. policy + Q over the chunk
        residual, combined_scaled, q_pred = _act_and_q(agent, obs)

        # 2. execute the whole chunk
        next_obs, reward, done, info = env.step_chunk(
            residual_chunk=residual, task_prompt=task_prompt, evaluation=True
        )

        # 3. collect xyz (whole chunk) + one Q per chunk
        combined_xyz = _to_xyz_steps(info["combined_action"])  # [C, 3] (unscaled/absolute)
        residual_xyz = _to_xyz_steps(info["residual_action"].reshape(env.chunk_len, -1))  # [C, 3]
        ep_combined.append(combined_xyz.copy())
        ep_residual.append(residual_xyz.copy())
        q_val = float(q_pred[0].item()) if q_pred.numel() else float(q_pred.mean().item())
        (ep_q_one if done_episodes < num_episodes else ep_q_two).append(q_val)

        # 4. episode end
        if bool(done.item()):
            is_success = bool(reward.item() > 0.0)
            progress_dots[done_episodes] = "✓" if is_success else "✗"
            logger.info(f"{task_prompt}: {progress_dots[done_episodes]}  {''.join(progress_dots)}")
            (successes_task_one if done_episodes < num_episodes else successes_task_two).append(is_success)
            successes.append(is_success)

            all_combined_trajs.append(
                np.concatenate(ep_combined, axis=0) if ep_combined else np.zeros((0, 3))
            )
            all_residual_trajs.append(
                np.concatenate(ep_residual, axis=0) if ep_residual else np.zeros((0, 3))
            )
            all_q_one.append(ep_q_one.copy())
            all_q_two.append(ep_q_two.copy())
            ep_combined, ep_residual, ep_q_one, ep_q_two = [], [], [], []

            done_episodes += 1
            if done_episodes == num_episodes:
                task_prompt = task_two
                if not switched:
                    logger.info("----------------------- switch to task two -------------------------")
                    switched = True
                next_obs = env.reset(task_prompt)

        obs = next_obs

    env.evaluation = False

    # 5. metrics
    sr_one = float(np.mean(successes_task_one)) if successes_task_one else 0.0
    sr_two = float(np.mean(successes_task_two)) if successes_task_two else 0.0
    flat_q_one = np.concatenate([np.asarray(t, np.float32) for t in all_q_one if t]) if any(all_q_one) else np.zeros((0,), np.float32)
    flat_q_two = np.concatenate([np.asarray(t, np.float32) for t in all_q_two if t]) if any(all_q_two) else np.zeros((0,), np.float32)

    metrics = {
        "eval/success_rate_task_one": sr_one,
        "eval/success_rate_task_two": sr_two,
        "eval/q_mean_task_one": float(flat_q_one.mean()) if flat_q_one.size else 0.0,
        "eval/q_mean_task_two": float(flat_q_two.mean()) if flat_q_two.size else 0.0,
    }

    # 6. plots -> wandb
    fig_combined = _plot_xyz_trajectories(all_combined_trajs, successes, title_prefix="Combined chunk", global_step=global_step)
    fig_residual = _plot_residual_per_dim(all_residual_trajs, successes, title_prefix="Residual chunk", global_step=global_step)
    fig_q_one = _plot_q_trajectories(all_q_one, successes, global_step=global_step) if save_q_plots else None
    fig_q_two = _plot_q_trajectories(all_q_two, successes, global_step=global_step) if save_q_plots else None

    if wandb.run is not None:
        log_dict = {
            **metrics,
            "eval/xyz_trajectories_combined": wandb.Image(fig_combined),
            "eval/xyz_trajectories_residual": wandb.Image(fig_residual),
        }
        if fig_q_one is not None:
            log_dict["eval/q_trajectories_task_one"] = wandb.Image(fig_q_one)
        if fig_q_two is not None:
            log_dict["eval/q_trajectories_task_two"] = wandb.Image(fig_q_two)
        wandb.log(log_dict, step=global_step)

    plt.close(fig_combined)
    plt.close(fig_residual)
    if fig_q_one is not None:
        plt.close(fig_q_one)
    if fig_q_two is not None:
        plt.close(fig_q_two)

    logger.info("----------------------------------------------------------------------")
    agent.train(True)
    return metrics
