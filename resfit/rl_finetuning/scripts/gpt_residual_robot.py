from __future__ import annotations

import contextlib
import dataclasses
import signal
from typing import Any, Optional
import copy
import cv2
import numpy as np
import requests
import torch
import json_numpy
from json_numpy import loads
from openpi_client import websocket_client_policy
import os
from datetime import datetime

from openpi_client import image_tools
from droid.franka_env import RobotEnv
json_numpy.patch()
from scipy.spatial.transform import Rotation as R
import time

from utils import process_policy_images, _extract_observation, prepare_image_256, to_hwc
from task_reward_generator import TaskRewardGenerator


DROID_CONTROL_FREQUENCY = 10  # 15

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

class Args:
    max_timesteps: int = 140 # 160 # 200 # 180   

    # GPT server(task_reward_generation_zedx.py)
    gpt_host: str = "127.0.0.1"  # 机器 IP
    gpt_port: int = 8007  # 端口

    # PI05 server(server.py)
    pi05_host: str = "127.0.0.1"  # 机器 IP
    pi05_port: int = 8008  # 端口



class BasePolicy:
    def __init__(self, main_host, main_port, action_scaler, state_standardizer, args=Args):
        self.device = torch.device("cuda") # cpu
        self.training = False
        self.server = f"http://{main_host}:{main_port}"
        self.action_scaler = action_scaler
        self._last_base_action = None
        self.state_standardizer = state_standardizer
        self.base_action_buffer =[]

        self.env = RobotEnv(action_space="cartesian_position", gripper_action_space="position")
        self.args = args
        self.pi05_client = websocket_client_policy.WebsocketClientPolicy(args.pi05_host, args.pi05_port)
        # self.text = "pick up the tomato and place it into the bowl"
        self.text = "pick up the cube and place it into the bowl"
        self.max_timesteps = args.max_timesteps
        self.t_step = 0
        self.round = 0
        self.reward = 0.0
        self.obs = None
        self.last_excution_time  = time.time()
        self.eefpose = None
        
        self.task_reward_generator = TaskRewardGenerator()

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
    
    def update_obs(self):
        self.obs = _extract_observation(
            self.env.get_observation(),
            save_to_disk=False,
        )
    def save_debug_images(self, obs_left, obs_right, obs_wrist, save_dir="debug_images"):
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        images = {
            "left": obs_left,
            "right": obs_right,
            "wrist": obs_wrist,
        }

        for name, img in images.items():
            path = os.path.join(save_dir, f"{timestamp}_{name}.png")

            # 如果是 RGB，cv2 保存前转成 BGR
            if img.ndim == 3 and img.shape[-1] == 3:
                img_to_save = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img_to_save = img
            cv2.imwrite(path, img_to_save)

    def get_obs_for_base(self):
        obs_left = copy.deepcopy(self.obs["left_image"])
        obs_right = copy.deepcopy(self.obs["right_image"])
        obs_wrist = copy.deepcopy(self.obs["wrist_image"])
        eef_pose = copy.deepcopy(self.obs["cartesian_position"]) # [x, y, z, roll, pitch, yaw]
        gripper_position = copy.deepcopy(self.obs["gripper_position"])
        
        self.eefpose = copy.deepcopy(eef_pose)

        eef_pos = eef_pose[:3]
        eef_rpy = eef_pose[3:6]  # [x, y, z, roll, pitch, yaw]
        eef_quat = R.from_euler('xyz', eef_rpy, degrees=False).as_quat()
        if eef_quat[3] < 0:  # 统一四元数符号，避免跳变
            eef_quat = -eef_quat
        eef_pose_quat = np.concatenate([eef_pos, eef_quat], axis=-1)

        obs_left = prepare_image_256(obs_left)
        obs_right = prepare_image_256(obs_right)
        obs_wrist = prepare_image_256(obs_wrist)
        self.save_debug_images(obs_left, obs_right, obs_wrist)

        return{
            "observation/exterior_image_1_left": image_tools.resize_with_pad(obs_left, 224, 224),
            "observation/wrist_image_left": image_tools.resize_with_pad(obs_right, 224, 224),
            "observation/exterior_image_2_left": image_tools.resize_with_pad(obs_wrist, 224, 224),
            "observation/eef_position": eef_pose_quat,
            "observation/gripper_position": gripper_position,
            "prompt": self.text,  # instruction
        }
    
    def get_obs_for_residual(self, base_naction):
        obs_left = torch.as_tensor(prepare_image_256(to_hwc(copy.deepcopy(self.obs["left_image"]))), dtype=torch.uint8, device=self.device)
        obs_right = torch.as_tensor(prepare_image_256(to_hwc(copy.deepcopy(self.obs["right_image"]))), dtype=torch.uint8, device=self.device)
        obs_wrist = torch.as_tensor(prepare_image_256(to_hwc(copy.deepcopy(self.obs["wrist_image"]))), dtype=torch.uint8, device=self.device)
        eef_pose = copy.deepcopy(np.asarray(self.obs["cartesian_position"], dtype=np.float32))
        gripper_position = self.obs["gripper_position"]
        
        eef_pos = eef_pose[:3]
        eef_rpy = eef_pose[3:6]
        eef_quat = R.from_euler('xyz', eef_rpy, degrees=False).as_quat()  # [qx, qy, qz, qw]
        if eef_quat[3] < 0:  # 统一四元数符号，避免跳变
            eef_quat = -eef_quat
        eef_pose_quat = np.concatenate([eef_pos, eef_quat], axis=-1)

        eef_pose_quat = torch.as_tensor(eef_pose_quat, dtype=torch.float32, device=self.device)
        gripper_position = torch.as_tensor(gripper_position, dtype=torch.float32, device=self.device)
        base_naction = torch.as_tensor(base_naction, dtype=torch.float32, device=self.device)

        if obs_left.ndim == 3:
            obs_left = obs_left.unsqueeze(0)
        if obs_right.ndim == 3:
            obs_right = obs_right.unsqueeze(0)
        if obs_wrist.ndim == 3:
            obs_wrist = obs_wrist.unsqueeze(0)

        if eef_pose_quat.ndim == 1:
            eef_pose_quat = eef_pose_quat.unsqueeze(0)
        if gripper_position.ndim == 1:
            gripper_position = gripper_position.unsqueeze(0)
        if base_naction.ndim == 1:
            base_naction = base_naction.unsqueeze(0)

        state = torch.cat([eef_pose_quat, gripper_position], dim=-1).to(self.device)

        obs = {
            "observation.state": state,
            "observation.base_action": base_naction,
            "observation.images.exterior_image_1_left": obs_left,
            "observation.images.exterior_image_2_left": obs_right,
            "observation.images.wrist_image_left": obs_wrist,
            "text": self.text,
        }
        augemnted_obs = self._augment_obs(obs, base_naction)

        return augemnted_obs

    def reset(self):
        self.t_step = 0
        self.env.reset()  # 1.593s

        self.pi05_client.reset()
        print("robot reset successfully")
        
        if self.round != 0:
            self.update_obs()
            self.reward =  float(self.task_reward_generator.reward_generation(self.obs["right_image"]))  # 0 or 1
            print("reward:", self.reward)
            time.sleep(7)

        print(f"---------------------------- trajectory {self.round} ----------------------------")
        self.update_obs()
        self.text, self.round = self.task_reward_generator.task_generation(self.obs["right_image"])  # 6.860s
        # print("task:", self.text)
        obs_for_pi0 = self.get_obs_for_base()
        action_base = self.get_online_action_base(obs_for_pi0)  # 0.093s

        # 标准化后的 base action
        self.base_action_buffer.clear()
        for action_ in action_base:
            self.base_action_buffer.append(action_)
        action_base_ = self.base_action_buffer.pop(0)
        base_naction = self.action_scaler.scale(action_base_)
        self._last_base_action = base_naction
        if base_naction.ndim == 1:
            base_naction = base_naction.unsqueeze(0)
        
        obs = self.get_obs_for_residual(base_naction)
        
        return obs

    def get_offline_action_base(self, raw_obs):
        if len(self.base_action_buffer)==0:
            query_action_base = True
        else:
            query_action_base = False

        obs_left = raw_obs["exterior_image_1_left"].detach().cpu().numpy()
        obs_right = raw_obs["wrist_image_left"].detach().cpu().numpy()
        obs_wrist = raw_obs["exterior_image_2_left"].detach().cpu().numpy()
        eef_pose = raw_obs["eef_position"].detach().cpu().numpy()  # [x, y, z, qx, qy, qz, qw]
        gripper_position = raw_obs["gripper_position"].detach().cpu().numpy()
        left_resized, right_resized, wrist_resized = process_policy_images(obs_left, obs_right, obs_wrist)
        
        if query_action_base:
            if eef_pose.ndim == 2 :
                eef_pose = eef_pose.squeeze(0)
            
            request_data = {
                "observation/exterior_image_1_left": image_tools.resize_with_pad(left_resized, 224, 224),
                "observation/wrist_image_left": image_tools.resize_with_pad(right_resized, 224, 224),
                "observation/exterior_image_2_left": image_tools.resize_with_pad(wrist_resized, 224, 224),
                "observation/eef_position": eef_pose,
                "observation/gripper_position": gripper_position,
                "prompt": self.text,  # instruction
            }

            pred_action_chunk = self.pi05_client.infer(request_data)["actions"]
            assert pred_action_chunk.shape == (50, 8) # 10,8
            # action = pred_action_chunk[0]
            action = pred_action_chunk[:35]  # 30 # 20
            # action = action[::2]

            action = np.asarray(action).copy()
            action[:, -1] = (action[:, -1] > 0.9).astype(action.dtype) # 0.6 0.5

            # 如果前3维是 delta position，就转成 absolute position
            action[:,:3] = action[:,:3] + eef_pose[:3]  # [20,8] todo!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            action_chunk = np.asarray(action, dtype=np.float32) 
        else:
            action_chunk = None

        if query_action_base:
            self.base_action_buffer.clear()
            for action in action_chunk:
                self.base_action_buffer.append(action)
        action_base = self.base_action_buffer.pop(0)

        return action_base
    
    def get_online_action_base(self, obs_for_pi0):
        pred_action_chunk = self.pi05_client.infer(obs_for_pi0)["actions"]
        assert pred_action_chunk.shape == (50, 8) # 10,8
        # action = pred_action_chunk[0]
        action = pred_action_chunk[:35]  # 30 # 20
        # action = action[::2]   # down sampling ferequency

        action = np.asarray(action).copy()
        action[:, -1] = (action[:, -1] > 0.9).astype(action.dtype) # 0.6 0.5
        # 如果前3维是 delta position，就转成 absolute position
        print(f"self.eefpose[:3] {self.eefpose[:3]}")
        action[:,:3] = action[:,:3] + self.eefpose[:3]  # [20,8] todo!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        action = np.asarray(action, dtype=np.float32) 
        action = torch.as_tensor(action, device=self.device)
        
        return action

    def step(self, residual_action):
        # residual_action = torch.zeros_like(residual_action) ##
        residual_action[:, -1] = 0
        combined_action = self._last_base_action + residual_action
        unscaled_combined_action = self.action_scaler.unscale(combined_action)

        if len(self.base_action_buffer)<1:
            query_action_base = True
        else:
            query_action_base = False
        
        # next_action_chunk, done, reward = self.get_transition(combined_action=unscaled_combined_action, query_action_base = query_action_base) # TODO: terminated, truncated?
        next_action_chunk, done = self.get_transition(combined_action=unscaled_combined_action, query_action_base = query_action_base) # TODO: terminated, truncated?
        
        if done:
            next_obs = self.reset()
            reward = self.reward
            print("reward: ", reward)

        else:
            reward = 0.0

            if query_action_base:
                self.base_action_buffer.clear()
                for action_ in next_action_chunk:
                    self.base_action_buffer.append(action_)
            
            next_base_action_ = self.base_action_buffer.pop(0)
            base_naction = self.action_scaler.scale(next_base_action_)
            self._last_base_action = base_naction  ##
            if base_naction.ndim == 1:
                base_naction = base_naction.unsqueeze(0)
            next_obs = self.get_obs_for_residual(base_naction)

            print("len(self.base_action_buffer)----------------------------->", len(self.base_action_buffer))
            print("next_base_action_  :", next_base_action_)
            print("residual_action    :", residual_action)
            print("combined_action    :", unscaled_combined_action)
        
        info = {}
        info["scaled_action"] = combined_action
        info["combined_action"] = unscaled_combined_action
        info["residual_action"] = residual_action

        reward = torch.as_tensor([reward], dtype=torch.float32, device=self.device)
        done = torch.as_tensor([done], dtype=torch.bool, device=self.device)

        return next_obs, reward, done, info

    def get_transition(self, combined_action, query_action_base):
        # 转成 numpy，并去掉 batch 维
        if isinstance(combined_action, torch.Tensor):
            combined_action = combined_action.detach().cpu().numpy()  # type: torch.Tensor
        combined_action = np.asarray(combined_action, dtype=np.float32)

        if combined_action.ndim == 2 and combined_action.shape[0] == 1:
            combined_action = combined_action[0]  # shape: torch.Size([1, 8]) -> (8,)
        assert combined_action.ndim == 1, f"Expected 1D combined_action, got shape {combined_action.shape}"

        #----------------------------quat -> rpy----------------------------#
        pos = combined_action[:3]
        q_action = combined_action[3:7]
        gripper = combined_action[7:]

        # 触地保护
        z_min = 0.22  # 0.225
        pos[2] = max(pos[2], z_min)
        
        norm = np.linalg.norm(q_action, keepdims=True)
        q_action = q_action / np.clip(norm, 1e-12, None)
        sign = np.where(q_action[..., 3:4] < 0, -1.0, 1.0)
        q_action = q_action * sign

        # if q_action[3] < 0:
        #     q_action = -q_action

        rpy_cmd = R.from_quat(q_action).as_euler('xyz', degrees=False)
        combined_action = np.concatenate([pos, rpy_cmd, gripper], axis=-1)
        # print("combined_action: ", combined_action)
        #-------------------------------------------------------------------#
        elapsed_time = time.time() - self.last_excution_time
        if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
            time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)
        
        self.env.step(combined_action)
        self.last_excution_time = time.time()
        self.t_step += 1
        self.update_obs() # update new obs
        print("trajectory steps:::::::::::::::::::::::", self.t_step)
        
        terminated = False  # terminated：任务本身的终止条件满足 TODO: 每隔一段时间请求一次 gpt 生成 reward
        truncated = (self.t_step >= self.args.max_timesteps - 1)  # truncated：被外部强制截断（时间上限等）
        done = terminated | truncated
        
        if done:
            # self.reset()
            # reward = self.reward
            done = done
            next_action_chunk = None
        else:
            if query_action_base:
                obs_for_pi0 = self.get_obs_for_base()
                next_action_chunk = self.get_online_action_base(obs_for_pi0)
            else:
                next_action_chunk = None
            # reward = 0.0
        
        return next_action_chunk, done # , reward
