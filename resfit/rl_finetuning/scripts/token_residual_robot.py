"""Chunk-form base policy that exposes the VLA "RL token".

Subclasses :class:`gpt_residual_robot.BasePolicy` for token-based, chunk-form
residual RL (RL-Token method).  Differences from the per-step base policy:

* Talks to the token-enabled PI0 server (``serve_policy_token.py``): every VLA
  query returns ``actions`` AND ``vla_tokens`` (prefix embeddings ``z_1:M``), which
  we cache in ``self.vla_tokens``.
* The RL observation is ``{observation.vla_tokens [M, emb],
  observation.state [1, S], observation.base_action [1, C*A]}`` where
  ``observation.base_action`` is the whole flattened, scaled VLA reference chunk.
* ``step_chunk(residual_chunk)`` executes all ``C`` steps of a chunk in one call
  (``combined = clamp(base + residual)`` per step), accumulates the discounted
  reward, and returns the next chunk's observation — i.e. one RL transition per
  VLA chunk.

Reward semantics are inherited unchanged (sparse; produced by the task reward
generator at episode reset).
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from gpt_residual_robot import DROID_CONTROL_FREQUENCY, Args, BasePolicy
from openpi_client import image_tools
from utils import process_policy_images
import time


class TokenBasePolicy(BasePolicy):
    def __init__(
        self,
        main_host,
        main_port,
        action_scaler,
        state_standardizer,
        *,
        chunk_len: int = 35,
        gamma: float = 0.99,
        args=Args,
    ):
        super().__init__(main_host, main_port, action_scaler, state_standardizer, args=args)
        self.chunk_len = int(chunk_len)
        self.gamma = float(gamma)
        self.vla_tokens: np.ndarray | None = None
        self._base_chunk_scaled: torch.Tensor | None = None  # [C, A] scaled

    # ------------------------------------------------------------------ #
    # VLA query (actions + tokens)                                        #
    # ------------------------------------------------------------------ #
    def _query_vla(self, task_prompt: str):
        """Return (base_chunk_unscaled [C, A] float32, vla_tokens [M, emb])."""
        obs_for_pi0 = self.get_obs_for_base(task_prompt)  # also sets self.eefpose
        resp = self.pi05_client.infer(obs_for_pi0)
        pred_action_chunk = resp["actions"]
        assert pred_action_chunk.shape[0] >= self.chunk_len, pred_action_chunk.shape
        vla_tokens = np.asarray(resp["vla_tokens"])  # [M, emb]

        action = np.asarray(pred_action_chunk[: self.chunk_len]).copy()
        action[:, -1] = (action[:, -1] > 0.9).astype(action.dtype)
        action[:, :3] = action[:, :3] + self.eefpose[:3]  # delta -> absolute position
        return action.astype(np.float32), vla_tokens

    def get_offline_tokens(self, raw_obs: dict, task_prompt: str):
        """Fresh per-frame VLA query for a demo frame (no rollout buffer).

        Returns (base_chunk_unscaled [C, A] float32, vla_tokens [M, emb]).  Used to
        pretrain the RL-token readout and to populate the offline replay buffer; the
        request is built exactly like ``BasePolicy.get_offline_action_base``.
        """
        obs_left = raw_obs["exterior_image_1_left"].detach().cpu().numpy()
        obs_right = raw_obs["wrist_image_left"].detach().cpu().numpy()
        obs_wrist = raw_obs["exterior_image_2_left"].detach().cpu().numpy()
        eef_pose = raw_obs["eef_position"].detach().cpu().numpy()  # [x,y,z,qx,qy,qz,qw]
        gripper_position = raw_obs["gripper_position"].detach().cpu().numpy()
        if eef_pose.ndim == 2:
            eef_pose = eef_pose.squeeze(0)

        left_resized, right_resized, wrist_resized = process_policy_images(obs_left, obs_right, obs_wrist)
        request_data = {
            "observation/exterior_image_1_left": image_tools.resize_with_pad(left_resized, 224, 224),
            "observation/wrist_image_left": image_tools.resize_with_pad(right_resized, 224, 224),
            "observation/exterior_image_2_left": image_tools.resize_with_pad(wrist_resized, 224, 224),
            "observation/eef_position": eef_pose,
            "observation/gripper_position": gripper_position,
            "prompt": task_prompt,
        }
        resp = self.pi05_client.infer(request_data)
        pred_action_chunk = resp["actions"]
        vla_tokens = np.asarray(resp["vla_tokens"])

        action = np.asarray(pred_action_chunk[: self.chunk_len]).copy()
        action[:, -1] = (action[:, -1] > 0.9).astype(action.dtype)
        action[:, :3] = action[:, :3] + eef_pose[:3]  # delta -> absolute position
        return action.astype(np.float32), vla_tokens

    def _scale_chunk(self, base_chunk_unscaled: np.ndarray) -> torch.Tensor:
        rows = [self.action_scaler.scale(torch.as_tensor(a, dtype=torch.float32, device=self.device))
                for a in base_chunk_unscaled]
        return torch.stack(rows, dim=0)  # [C, A]

    def _current_state(self) -> torch.Tensor:
        """Standardized proprio state [1, S] from the latest observation."""
        eef_pose = copy.deepcopy(np.asarray(self.obs["cartesian_position"], dtype=np.float32))
        eef_pos = eef_pose[:3]
        eef_quat = R.from_euler("xyz", eef_pose[3:6], degrees=False).as_quat()
        if eef_quat[3] < 0:
            eef_quat = -eef_quat
        eef_pose_quat = np.concatenate([eef_pos, eef_quat], axis=-1)
        eef_pose_quat = torch.as_tensor(eef_pose_quat, dtype=torch.float32, device=self.device)
        gripper = torch.as_tensor(self.obs["gripper_position"], dtype=torch.float32, device=self.device)
        if eef_pose_quat.ndim == 1:
            eef_pose_quat = eef_pose_quat.unsqueeze(0)
        if gripper.ndim == 1:
            gripper = gripper.unsqueeze(0)
        state = torch.cat([eef_pose_quat, gripper], dim=-1)
        return self.state_standardizer.standardize(state)

    def _token_obs(self) -> dict[str, torch.Tensor]:
        assert self._base_chunk_scaled is not None and self.vla_tokens is not None
        return {
            "observation.state": self._current_state(),  # [1, S]
            "observation.base_action": self._base_chunk_scaled.reshape(1, -1),  # [1, C*A]
            "observation.vla_tokens": torch.as_tensor(
                self.vla_tokens, dtype=torch.float32, device=self.device
            ),  # [M, emb]
        }

    # ------------------------------------------------------------------ #
    # Episode reset                                                      #
    # ------------------------------------------------------------------ #
    def reset(self, task_prompt=None):  # type: ignore[override]
        self.t_step = 0
        self.env.reset()
        self.pi05_client.reset()
        print("robot reset successfully")

        if self.round != 0:
            self.update_obs()
            self.reward = float(self.task_reward_generator.reward_generation(self.obs["right_image"], task_prompt))
            time.sleep(7)

        print(f"---------------------------- trajectory {self.round} ----------------------------")
        self.update_obs()
        if self.evaluation is False:
            self.text, self.round = self.task_reward_generator.task_generation(self.obs["right_image"])
        print("current task:", self.text)

        prompt = self.text if task_prompt is None else task_prompt
        base_chunk_unscaled, self.vla_tokens = self._query_vla(prompt)
        self._base_chunk_scaled = self._scale_chunk(base_chunk_unscaled)

        obs = self._token_obs()
        if task_prompt is None:
            return obs, self.text
        return obs

    # ------------------------------------------------------------------ #
    # Chunk step                                                         #
    # ------------------------------------------------------------------ #
    def _apply_action(self, combined_unscaled_row: np.ndarray) -> bool:
        """Execute a single unscaled action row on the robot. Returns done."""
        combined = np.asarray(combined_unscaled_row, dtype=np.float32).reshape(-1)
        pos = combined[:3]
        q_action = combined[3:7]
        gripper = combined[7:]

        pos[2] = max(pos[2], 0.22)  # table protection
        norm = np.linalg.norm(q_action, keepdims=True)
        q_action = q_action / np.clip(norm, 1e-12, None)
        q_action = q_action * np.where(q_action[..., 3:4] < 0, -1.0, 1.0)
        rpy_cmd = R.from_quat(q_action).as_euler("xyz", degrees=False)
        cmd = np.concatenate([pos, rpy_cmd, gripper], axis=-1)

        elapsed = time.time() - self.last_excution_time
        if elapsed < 1 / DROID_CONTROL_FREQUENCY:
            time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed)
        self.env.step(cmd)
        self.last_excution_time = time.time()
        self.t_step += 1
        self.update_obs()

        truncated = self.t_step >= self.args.max_timesteps - 1
        if truncated:
            cmd[2] += 0.1  # lift at episode end
            self.env.step(cmd)
        return bool(truncated)

    def step_chunk(self, residual_chunk, task_prompt=None, evaluation=False):
        """Execute a full chunk. ``residual_chunk`` is [C, A] or [1, C*A] (scaled)."""
        self.evaluation = evaluation
        A = self._base_chunk_scaled.shape[-1]
        residual = torch.as_tensor(residual_chunk, dtype=torch.float32, device=self.device).reshape(self.chunk_len, A)
        residual[:, -1] = 0.0  # no residual on the gripper (matches per-step base policy)

        combined_scaled = self._base_chunk_scaled + residual  # [C, A] (still scaled)
        combined_unscaled = torch.stack(
            [self.action_scaler.unscale(combined_scaled[i]) for i in range(self.chunk_len)], dim=0
        ).detach().cpu().numpy()

        discounted_reward = 0.0
        done = False
        steps_taken = 0
        for i in range(self.chunk_len):
            done = self._apply_action(combined_unscaled[i])
            steps_taken += 1
            if done:
                break
        print(f"[chunk] executed {steps_taken}/{self.chunk_len} steps, t_step={self.t_step}, done={done}")

        info = {
            "scaled_action": combined_scaled.reshape(1, -1),  # [1, C*A] executed chunk action (scaled)
            "combined_action": combined_unscaled,  # [C, A] executed chunk action (unscaled, absolute)
            "residual_action": residual.reshape(1, -1),
            "steps_taken": steps_taken,
            "task_prompt": self.text,
        }

        if done:
            if task_prompt is None:
                next_obs, _ = self.reset()
            else:
                next_obs = self.reset(task_prompt=task_prompt)
            # Sparse terminal reward, discounted to the step where the episode ended.
            discounted_reward = (self.gamma ** (steps_taken - 1)) * float(self.reward)
        else:
            base_chunk_unscaled, self.vla_tokens = self._query_vla(task_prompt or self.text)
            self._base_chunk_scaled = self._scale_chunk(base_chunk_unscaled)
            next_obs = self._token_obs()

        reward = torch.as_tensor([discounted_reward], dtype=torch.float32, device=self.device)
        done_t = torch.as_tensor([done], dtype=torch.bool, device=self.device)
        return next_obs, reward, done_t, info
