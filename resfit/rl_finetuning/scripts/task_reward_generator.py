import os
import sys

import cv2
import torch
import time, threading, queue, random
import subprocess
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from datetime import datetime
from pathlib import Path

import pyzed.sl as sl
import base64
from dotenv import load_dotenv
from openai import OpenAI
from groundingdino.util.inference import load_model, load_image, predict, annotate
from loguru import logger

import uvicorn
import zerorpc
import asyncio
import json
from typing import Dict, Any, Optional, Tuple
import numpy as np

import json_numpy
json_numpy.patch()

from zedx_streamer import ZEDStreamer


class TaskRewardGenerator:
    def __init__(
        self,
        max_timesteps: int = 60,
        zed_stream_ip: str = "192.168.55.1",
        zed_stream_port: int = 30000,
    ):
        # Configuration
        self.max_timesteps = max_timesteps
        self.current_scene_gdino = None
        self.next_scene_gdino = None

        # camera
        self.zed = None
        # self.camera = None
        self.camera = ZEDStreamer()
        self.camera.start(stream_ip=zed_stream_ip, stream_port=zed_stream_port)
        self.image = None

        # Runtime state
        self.round_current = 0
        self.round_next = 1
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.queue: "Queue[tuple[list[str], list[str]]]" = Queue()

        # Hardware / model setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = load_model(
            "/home/yuan/self_vla/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            "/home/yuan/self_vla/GroundingDINO/weights/groundingdino_swint_ogc.pth",
        ).to(self.device).eval()
        self.runtime_parameters = None

        # OpenAI client
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Logging
        self.setup_logger()

        self.round = 0
        self.scene_before = None
        self.scene_after = None
        self.selected_task = None
        self.selected_objects = None
        self.img_rgb = None

        self.candidate_tasks = [
            "pick up the tomato and place it into the bowl"
            "pick up the tomato from the bowl and place it in front of the bowl"
            "pick up the tomato from the bowl and place it behind the bowl"
            "pick up the tomato from the bowl and place it to the left of the bowl"
            "pick up the tomato from the bowl and place it to the right of the bowl"
        ]

    def setup_logger(self):
        log_dir = f"outputs/task_reward_generation/{self.timestamp}"
        log_path = os.path.join(log_dir, "log.txt")

        logger.remove()
        logger.add(log_path, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO")
        logger.add(sys.stdout, colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
        # logger.add(lambda msg: print(msg, end=""), format="{message}")

    def mask_image(self, image):
        """
        Args:
            image: np.ndarray, 原图, shape=(H, W, 3)
            polygon_points: list 或 np.ndarray
                T 字形区域顶点，按顺时针或逆时针排列
                例如:
                [
                    [x1, y1],
                    [x2, y2],
                    ...
                ]
            background_color: tuple
                非桌面区域填充值，默认黑色 (0, 0, 0)

        Returns:
            mask: np.ndarray, shape=(H, W), dtype=np.uint8
                桌面区域为 255,其余区域为 0
            masked_image: np.ndarray, shape=(H, W, 3)
                仅保留桌面区域后的图像
        """
        polygon_points = [
            [1500, 150],  # 右上
            [1600, 545],  # 右1/2外
            [1310, 535],  # 右1/2内
            [1320, 715],  # 右1/4外
            [1280, 715],  # 右1/4内
            [1285, 830],  # 右下

            [600, 820],   # 左下
            [615, 715],   # 左1/4内
            [575, 715],   # 左1/4外
            [605, 530],   # 左1/2内
            [345, 520],   # 左1/2外
            [460, 135],   # 左上
        ]
        background_color=(0, 0, 0)

        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        polygon = np.array(polygon_points, dtype=np.int32)
        cv2.fillPoly(mask, [polygon], 255)

        masked_image = np.full_like(image, background_color, dtype=image.dtype)
        masked_image[mask > 0] = image[mask > 0]

        return masked_image

    def process_img(self, img_rgb):
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # save_dir = f"output/task_reward_generation/{self.timestamp}"
        # os.makedirs(save_dir, exist_ok=True)
        # save_path = os.path.join(save_dir, f"scene_{round}.jpg")
        # # cv2.imwrite(save_path, img_bgr)
        # cv2.imwrite(save_path, img_rgb)

        ok, buffer = cv2.imencode(".jpg", img_bgr)
        if ok:
            b64jpg = base64.b64encode(buffer.tobytes()).decode("utf-8")
        else:
            raise RuntimeError("Failed to encode image to JPEG")
        
        # masked_img_rgb = self.mask_image(img_rgb)
        masked_img_bgr = self.mask_image(img_bgr)

        # return masked_img_rgb, b64jpg
        return masked_img_bgr, b64jpg

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def gdino(self, scene, before):
        # IMAGE_PATH = image_path # "weights/dog-3.jpeg"
        TEXT_PROMPT = self.selected_objects # "chair . person . dog ."
        BOX_TRESHOLD = 0.32 # 0.35
        TEXT_TRESHOLD = 0.25 # 0.25

        image_source, image = load_image(scene)  # IMAGE_PATH

        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )

        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

        save_dir = f"outputs/task_reward_generation/{self.timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        if before == True:
            save_path = os.path.join(save_dir, f"0_annotated_image_{self.round}_before.jpg")
        else:
            save_path = os.path.join(save_dir, f"1_annotated_image_{self.round}_after.jpg")
        cv2.imwrite(save_path, annotated_frame)

        scene_gdino = self.encode_image(save_path)

        return scene_gdino

    def task_generator(self, scene, candidate_tasks):
        tasks_text = "\n".join([f"- {task}" for task in candidate_tasks])

        prompt_task = (
            "You are a one-arm robot. Based on the image, judge the spatial relationships between objects.\n"
            "From the task list below, randomly choose ONE feasible task for the current scene.\n"
            "Do not invent or modify tasks.\n\n"
            f"Task list:\n{tasks_text}\n\n"
            "Return exactly in this format:\n"
            "task: <selected task>\n"
            "objects: object1 . object2 .\n"
        )

        response = self.client.responses.create(
            model="gpt-4.1-mini", # gpt-4.1-mini
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt_task},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{scene}",},
                ],
            }],
        )

        output = response.output_text
        
        selected_task = None
        selected_objects = None

        for line in output.splitlines():
            line = line.strip()

            if line.lower().startswith("task:"):
                selected_task = line.split(":", 1)[1].strip()

            elif line.lower().startswith("objects:"):
                selected_objects = line.split(":", 1)[1].strip()

        self.selected_task = selected_task
        self.selected_objects = selected_objects

    def reward_generator(self, current_scene_gdino, next_scene_gdino, current_task):   
        # image_path = "/home/yuan/self_vla/residual-offpolicy-rl/outputs/task_reward_generation/20260426_205142_/annotated_image_15.jpg"
        # current_scene_gdino = self.encode_image(image_path)
        # save_dir = f"/home/yuan/self_vla/residual-offpolicy-rl/outputs/task_reward_generation/initial_scene"
        # os.makedirs(save_dir, exist_ok=True)
        # save_path = os.path.join(save_dir, f"initial_scene_{self.round}.jpg")
        # image_bytes = base64.b64decode(current_scene_gdino)
        # with open(save_path, "wb") as f:
        #     f.write(image_bytes)

        prompt_reward = (
            "You are given two images:\n"
            "- The **first image** shows the initial scene **before** the robot starts the task.\n"
            "- The **second image** shows the result **after** the robot attempted the task.\n\n"

            "The robot was instructed to perform the following task:\n"
            f"{current_task}\n\n"

            "Instructions:\n"
            "1. From the camera's perspective, carefully look at the **initial position** of the key object(s) mentioned in the task.\n"
            "   - The top of the image represents the **front** of the object.\n"
            "   - The bottom of the image represents the **behind** of the object.\n"
            "   - The left side of the image represents the **left side** of the object.\n"
            "   - The right side of the image represents the **right side** of the object.\n"
            "2. Pay attention to whether the object is inside something like a bowl or container.\n"
            "3. Then carefully look at the **final position** of the key object in the second image.\n"
            "4. **VERY IMPORTANT: If bounding boxes are visible, focus on the exact **box labels** — make sure you refer to the correct object name!**\n"
            "   - Do NOT confuse objects with similar color/shape.\n"
            "   - If labels clearly show containment or alignment, include that in your reasoning.\n"
            "5. Compare the final position with the task requirement. "

            "**Respond with only a single digit: `1` if the task was successfully completed, or `0` if it failed.**\n"
            "**Do not give a reason**, just output a single digit:\n"
            "If the key object(s) are placed exactly as instructed, output `1`.\n"
            "If not (wrong position, ambiguous, or missing), output `0`.\n\n"
        )   

        response = self.client.responses.create(
            model="gpt-4.1-mini", # gpt-4.1-mini
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt_reward},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{current_scene_gdino}",},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{next_scene_gdino}",},
                ],
            }],
        )

        reward = response.output_text

        return reward

    # task_0-(reward_0-task_1)-(reward_1-task_2)-...
    def task_generation(self, img_rgb):  # shape: (1080, 1920, 3) dtype: uint8
        # if img_rgb is not None:  # round=0
        #     scene_dino, scene_gpt = self.process_img(img_rgb)  # img_rgb
        #     self.task_generator(scene_gpt, self.candidate_tasks)

        #     self.scene_before = self.gdino(scene_dino, self.round)
        # else:  # round=1,2,...
        #     scene_dino, scene_gpt = self.process_img(self.img_rgb)  # self.img_rgb
        #     self.task_generator(scene_gpt, self.candidate_tasks)

        scene_dino, scene_gpt = self.process_img(img_rgb)
        self.task_generator(scene_gpt, self.candidate_tasks)
        
        self.round += 1
        self.scene_before = self.gdino(scene_dino, before=True)

        logger.info(f"Selected task to execute: {self.selected_task}")
        logger.info(f"Selected objects to detect: {self.selected_objects}")
        
        return self.selected_task, self.round
    
    def reward_generation(self, img_rgb):
        next_scene_dino, next_scene_gpt = self.process_img(img_rgb)
        self.scene_after = self.gdino(next_scene_dino, before=False)

        reward = self.reward_generator(self.scene_before, self.scene_after, self.selected_task)
        
        # self.round += 1
        # self.img_rgb = img_rgb
        # self.scene_before = self.scene_after
        
        return reward

# just for test
# if __name__ == '__main__':
#     # image_path = "/home/yuan/self_vla/outputs/task_reward_generation/20260419_153546/annotated_image_21.jpg"
#     image_path = "/home/yuan/self_vla/outputs/task_reward_generation/20260419_150621/annotated_image_0.jpg"

#     img_bgr = cv2.imread(image_path)
#     if img_bgr is None:
#         raise FileNotFoundError(f"Failed to read image: {image_path}")

#     img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

#     task_reward_generator = TaskRewardGenerator()
#     selected_task, round_id = task_reward_generator.task_generation(img_rgb)


