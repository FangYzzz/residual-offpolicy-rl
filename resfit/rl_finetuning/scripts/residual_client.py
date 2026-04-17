from __future__ import annotations

import contextlib
import dataclasses
import signal
from typing import Any, Optional

import cv2
import numpy as np
import requests
import torch
import json_numpy
from json_numpy import loads
from openpi_client import image_tools

json_numpy.patch()


@contextlib.contextmanager
def prevent_keyboard_interrupt():
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


# def prepare_image_256(img: np.ndarray, size=(256, 256)) -> np.ndarray:
#     """中心裁剪成正方形，再缩放到指定大小，输出 RGB uint8。"""
#     if img is None:
#         raise ValueError("Input image is None")
#     h, w = img.shape[:2]
#     side = min(h, w)
#     y0 = (h - side) // 2
#     x0 = (w - side) // 2
#     crop = img[y0:y0 + side, x0:x0 + side]
#     out = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
#     return out.astype(np.uint8, copy=False)


# def process_policy_images_from_obs(
#     obs: dict[str, Any],
#     external_camera_key: str = "observation.images.agentview",
#     wrist_camera_key: str = "observation.images.robot0_eye_in_left_hand",
# ) -> tuple[np.ndarray, np.ndarray]:
#     """
#     输入训练/评估里常见的 observation dict，输出 Flower server 需要的两张 224x224 图。
#     """
#     primary_image = obs[external_camera_key]
#     wrist_image = obs[wrist_camera_key]

#     if isinstance(primary_image, torch.Tensor):
#         primary_image = primary_image.detach().cpu().numpy()
#     if isinstance(wrist_image, torch.Tensor):
#         wrist_image = wrist_image.detach().cpu().numpy()

#     primary_image = np.asarray(primary_image)
#     wrist_image = np.asarray(wrist_image)

#     # 支持 CHW -> HWC
#     if primary_image.ndim == 3 and primary_image.shape[0] in (1, 3):
#         primary_image = np.transpose(primary_image, (1, 2, 0))
#     if wrist_image.ndim == 3 and wrist_image.shape[0] in (1, 3):
#         wrist_image = np.transpose(wrist_image, (1, 2, 0))

#     primary_image = prepare_image_256(primary_image)
#     wrist_image = prepare_image_256(wrist_image)

#     primary_resized = image_tools.resize_with_pad(primary_image, 224, 224)
#     wrist_resized = image_tools.resize_with_pad(wrist_image, 224, 224)

#     return primary_resized, wrist_resized


class ResidualClient:
    def __init__(self, main_host, main_port, action_scaler, state_standardizer):
        self.device = torch.device("cuda") # cpu
        self.training = False
        self.server = f"http://{main_host}:{main_port}"
        self.action_scaler = action_scaler
        self._last_base_action = None
        self.state_standardizer = state_standardizer
        self.base_action_buffer =[]

    def to(self, device: str | torch.device):
        self.device = torch.device(device)
        return self

    def eval(self):
        self.training = False
        return self

    def train(self, mode: bool = True):
        self.training = mode
        return self
    
    def _augment_obs(self, raw_obs: dict[str, torch.Tensor], base_naction: torch.Tensor) -> dict[str, torch.Tensor]:
        """Augment observations with base actions."""

        # New way to do this is to just add the base action to the state under its own key
        augmented_obs = raw_obs.copy()
        augmented_obs["observation.base_action"] = base_naction
        augmented_obs["observation.state"] = self.state_standardizer.standardize(augmented_obs["observation.state"])

        return augmented_obs

    def reset(self):
        with prevent_keyboard_interrupt():
            resp = requests.post(
                f"{self.server}/reset",
                json={},
                timeout=60,
            )
        resp.raise_for_status()
        raw = resp.json()

        # raw -> tensor
        obs_left = torch.as_tensor(np.array(raw["obs_left"]), device=self.device)
        obs_right = torch.as_tensor(np.array(raw["obs_right"]), device=self.device)
        obs_wrist = torch.as_tensor(np.array(raw["obs_wrist"]), device=self.device)

        eef_position = torch.as_tensor(np.array(raw["eef_position"]), device=self.device, dtype=torch.float32)
        gripper_position = torch.as_tensor(np.array(raw["gripper_position"]), device=self.device, dtype=torch.float32)
        action_base = torch.as_tensor(np.array(raw["action_base"]), device=self.device, dtype=torch.float32)  # todo!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!命名统一

        text = raw["text"]

        # num_envs == 1，但保持多环境接口格式
        if obs_left.ndim == 3:
            obs_left = obs_left.unsqueeze(0)
        if obs_right.ndim == 3:
            obs_right = obs_right.unsqueeze(0)
        if obs_wrist.ndim == 3:
            obs_wrist = obs_wrist.unsqueeze(0)

        if eef_position.ndim == 1:
            eef_position = eef_position.unsqueeze(0)
        if gripper_position.ndim == 1:
            gripper_position = gripper_position.unsqueeze(0)
        if action_base.ndim == 1:
            action_base = action_base.unsqueeze(0)

        # 标准化后的 base action
        self.base_action_buffer.clear()
        for action_ in action_base:
            self.base_action_buffer.append(action_)
        action_base_ = self.base_action_buffer.pop(0)
        base_naction = self.action_scaler.scale(action_base_)
        self._last_base_action = base_naction

        # 与 offline schema 对齐
        state = torch.cat([eef_position, gripper_position], dim=-1)

        obs = {
            "observation.state": state,
            "observation.base_action": base_naction,
            "exterior_image_1_left": obs_left,
            "exterior_image_2_left": obs_right,
            "wrist_image_left": obs_wrist,
            "text": text,
        }
        if base_naction.ndim == 1:
            base_naction = base_naction.unsqueeze(0)

        obs = self._augment_obs(obs, base_naction)
        return obs

    def get_curr_obs(self):
        with prevent_keyboard_interrupt():
            resp = requests.post(
                f"{self.server}/query_curr_obs",
                json={},
                # timeout=1.0,
            )
        obs = np.array(loads(resp.json()))
        return obs

    def get_offline_action_base(self, raw_obs):
        if len(self.base_action_buffer):
            query_action_base = True
        else:
            query_action_base = False
        with prevent_keyboard_interrupt():
            resp = requests.post(
                f"{self.server}/query_offline_action_base",
                json={
                    "exterior_image_1_left": raw_obs["exterior_image_1_left"].detach().cpu().numpy(),
                    "wrist_image_left": raw_obs["wrist_image_left"].detach().cpu().numpy(),
                    "exterior_image_2_left": raw_obs["exterior_image_2_left"].detach().cpu().numpy(),
                    "eef_position": raw_obs["eef_position"].detach().cpu().numpy(),
                    "gripper_position": raw_obs["gripper_position"].detach().cpu().numpy(),
                    "query_action_base": query_action_base,
                },
                # timeout=1.0,
            )

        if query_action_base:
            self.base_action_buffer.clear()
            action_base_chunk = torch.asarray(loads(resp.json()))
            for action in action_base_chunk:
                self.base_action_buffer.append(action)
        action_base = self.base_action_buffer.pop(0)

        return action_base
    
    # def get_online_action_base(self):
    #     with prevent_keyboard_interrupt():
    #         resp = requests.post(
    #             f"{self.server}/query_online_action_base",
    #             json={},
    #             # timeout=1.0,
    #         )
        
    #     action_base = torch.asarray(loads(resp.json()))
    #     return action_base
    
    def step(self, residual_action):
        # residual_action = torch.zeros_like(residual_action)
        residual_action[:, -1] = 0
        combined_action = self._last_base_action + residual_action
        unscaled_combined_action = self.action_scaler.unscale(combined_action)

        if len(self.base_action_buffer)==0:
            query_action_base = True
        else:
            query_action_base = False
        
        next_obs, reward, done = self.get_transition(combined_action=unscaled_combined_action, query_action_base = query_action_base) # TODO: terminated, truncated?

        # next_base_action = next_obs["observation.base_action"]
        if query_action_base or done:
            next_base_action = next_obs["observation.base_action"]
            self.base_action_buffer.clear()
            for action_ in next_base_action:
                self.base_action_buffer.append(action_)
        
        next_base_action_ = self.base_action_buffer.pop(0)
        base_naction = self.action_scaler.scale(next_base_action_)
        self._last_base_action = base_naction
        if base_naction.ndim == 1:
            base_naction = base_naction.unsqueeze(0)
        
        next_obs = self._augment_obs(next_obs, base_naction)

        info = {}
        info["scaled_action"] = combined_action
        # info = {
        #     "scaled_action": combined_action,
        #     "episode_steps": ,
        #     "_episode_steps": ,
        # }

        print("len(self.base_action_buffer)----------------------------->", len(self.base_action_buffer))
        print("next_base_action_  :", next_base_action_)
        print("residual_action    :", residual_action)
        print("combined_action    :", combined_action)

        return next_obs, reward, done, info

    def get_transition(self, combined_action, query_action_base):
        # 转成 numpy，并去掉 batch 维
        if isinstance(combined_action, torch.Tensor):
            combined_action = combined_action.detach().cpu().numpy()  # type: torch.Tensor
        combined_action = np.asarray(combined_action, dtype=np.float32)

        if combined_action.ndim == 2 and combined_action.shape[0] == 1:
            combined_action = combined_action[0]  # shape: torch.Size([1, 8]) -> (8,)
        assert combined_action.ndim == 1, f"Expected 1D combined_action, got shape {combined_action.shape}"

        with prevent_keyboard_interrupt():
            resp = requests.post(
                f"{self.server}/query_transition",
                json={
                    "combined_action": combined_action.tolist(),
                    "query_action_base": query_action_base,
                },
                timeout=60,
            )
        resp.raise_for_status()
        raw = resp.json()

        # 原始数据 -> tensor
        obs_left = torch.as_tensor(np.array(raw["obs_left"]), device=self.device)
        obs_right = torch.as_tensor(np.array(raw["obs_right"]), device=self.device)
        obs_wrist = torch.as_tensor(np.array(raw["obs_wrist"]), device=self.device)

        eef_position = torch.as_tensor(np.array(raw["eef_position"]), device=self.device, dtype=torch.float32)
        gripper_position = torch.as_tensor(np.array(raw["gripper_position"]), device=self.device, dtype=torch.float32)
        if raw["next_action_base"] is not None:
            next_base_action = torch.as_tensor(np.array(raw["next_action_base"]), device=self.device, dtype=torch.float32)
        else:
            next_base_action = None

        reward = torch.as_tensor(raw["reward"], device=self.device, dtype=torch.float32)
        done = torch.as_tensor(raw["done"], device=self.device, dtype=torch.bool)

        # 保证都有 env 维
        if eef_position.ndim == 1:
            eef_position = eef_position.unsqueeze(0)          # [D] -> [1, D]
        if gripper_position.ndim == 1:
            gripper_position = gripper_position.unsqueeze(0)  # [D] -> [1, D]
        if next_base_action is not None:
            if next_base_action.ndim == 1:
                next_base_action = next_base_action.unsqueeze(0)  # [A] -> [1, A]

        if obs_left.ndim == 3:
            obs_left = obs_left.unsqueeze(0)      # [H, W, C] -> [1, H, W, C]
        if obs_right.ndim == 3:
            obs_right = obs_right.unsqueeze(0)
        if obs_wrist.ndim == 3:
            obs_wrist = obs_wrist.unsqueeze(0)

        if reward.ndim == 0:
            reward = reward.unsqueeze(0)          # [] -> [1]
        if done.ndim == 0:
            done = done.unsqueeze(0)              # [] -> [1]

        # 和 offline key 对齐
        state = torch.cat([eef_position, gripper_position], dim=-1)   # [1, D_state]

        next_obs = {
            "observation.state": state,
            "observation.base_action": next_base_action,
            "exterior_image_1_left": obs_left,  # torch.Size([1, 224, 224, 3])
            "exterior_image_2_left": obs_right,
            "wrist_image_left": obs_wrist,
        }

        return next_obs, reward, done
