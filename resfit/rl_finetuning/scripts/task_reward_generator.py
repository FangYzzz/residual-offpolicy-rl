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
        # self.camera = ZEDStreamer(exposure=40, gain=50, auto_exposure=False)
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
            ### task1 ###
            # cube_in50_out50
            # "pick up the cube and place it into the bowl",
            # "pick up the cube from the bowl and place it outside the bowl",
            # cube_fix_in80_out50
            "put the cube into the bowl",
            "put the cube outside the bowl",
            ### task2 ###
            # "open the drawer"
            # "close the drawer"
        ]

    def setup_logger(self):
        log_dir = f"outputs/task_reward_generation/{self.timestamp}"
        log_path = os.path.join(log_dir, "log.txt")

        logger.remove()
        logger.add(log_path, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO")
        logger.add(sys.stdout, colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
        # logger.add(lambda msg: print(msg, end=""), format="{message}")

    def reduce_exposure_bgr(self, img_bgr, alpha=0.65, beta=-20):
        """
        只用于 GPT / DINO 的图像预处理。
        new_pixel = alpha * old_pixel + beta
        alpha < 1: 降低整体亮度/对比度，亮的地方不那么亮
        beta < 0: 整体压暗
        """
        return cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta)
    
    # def enhance_bowl_visibility_bgr(self, img_bgr):
    #     """
    #     更推荐版本：
    #     只在低饱和、高亮区域附近增强，避免增强整张地面。
    #     """

    #     hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    #     h, s, v = cv2.split(hsv)

    #     # 初步找白色/灰白色物体区域
    #     candidate_mask = ((v > 130) & (s < 80)).astype(np.uint8) * 255

    #     kernel = np.ones((7, 7), np.uint8)

    #     # 去噪
    #     candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, kernel)

    #     # 扩大一点，让 bowl 边缘也被处理到
    #     candidate_mask = cv2.dilate(candidate_mask, kernel, iterations=2)

    #     # 平滑 mask
    #     candidate_mask = cv2.GaussianBlur(candidate_mask, (31, 31), 0)

    #     mask_f = candidate_mask.astype(np.float32) / 255.0
    #     mask_f = mask_f[..., None]

    #     # 对整图生成一个增强版本
    #     lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    #     l, a, b = cv2.split(lab)

    #     clahe = cv2.createCLAHE(
    #         clipLimit=1.5,
    #         tileGridSize=(8, 8)
    #     )
    #     l_clahe = clahe.apply(l)

    #     lab_clahe = cv2.merge([l_clahe, a, b])
    #     img_enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    #     # 再轻微压暗增强图中的高亮
    #     hsv_enhanced = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2HSV)
    #     h2, s2, v2 = cv2.split(hsv_enhanced)

    #     v2 = np.where((v2 > 180) & (s2 < 100), v2 * 0.8, v2)
    #     v2 = np.clip(v2, 0, 255).astype(np.uint8)

    #     hsv_enhanced = cv2.merge([h2, s2, v2])
    #     img_enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

    #     # 只在候选 bowl 区域融合增强图
    #     out = img_bgr.astype(np.float32) * (1.0 - mask_f) + img_enhanced.astype(np.float32) * mask_f

    #     return np.clip(out, 0, 255).astype(np.uint8)
    
    # def enhance_cube_visibility_bgr(self, img_bgr):
    #     """
    #     专门增强 cube 的颜色和边缘：
    #     1. 找到有颜色的区域，例如绿色/橙色 cube
    #     2. 只增强这些区域的饱和度
    #     3. 对这些区域做轻微锐化，让边缘更清晰
    #     """

    #     img_float = img_bgr.astype(np.float32)

    #     # BGR -> HSV，方便增强颜色
    #     hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    #     h, s, v = cv2.split(hsv)

    #     # 找有颜色的区域
    #     # cube 是绿色/橙色，饱和度会比 bowl 和地面高
    #     color_mask = ((s > 35) & (v > 60)).astype(np.uint8) * 255

    #     # 去掉小噪点
    #     kernel = np.ones((3, 3), np.uint8)
    #     color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)

    #     # 扩大一点，覆盖 cube 边缘
    #     color_mask = cv2.dilate(color_mask, kernel, iterations=1)

    #     # 平滑 mask，避免边缘突兀
    #     color_mask = cv2.GaussianBlur(color_mask, (9, 9), 0)

    #     mask_f = color_mask.astype(np.float32) / 255.0
    #     mask_f = mask_f[..., None]

    #     # 增强饱和度和亮度
    #     s_enhanced = np.clip(s * 2.0, 0, 255)
    #     v_enhanced = np.clip(v * 1.08, 0, 255)

    #     hsv_enhanced = cv2.merge([h, s_enhanced, v_enhanced]).astype(np.uint8)
    #     img_color_enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

    #     # 轻微锐化，增强 cube 边界
    #     blur = cv2.GaussianBlur(img_color_enhanced, (0, 0), 1.0)
    #     img_sharp = cv2.addWeighted(img_color_enhanced, 1.6, blur, -0.6, 0)

    #     # 只在彩色区域融合增强结果
    #     out = img_float * (1.0 - mask_f) + img_sharp.astype(np.float32) * mask_f

    #     return np.clip(out, 0, 255).astype(np.uint8)

    def enhance_cube_visibility_bgr(self, img_bgr):
        """
        专门增强 cube 的颜色和边缘：
        1. 找到有颜色的区域，例如绿色/橙色/淡黄色 cube
        2. 只增强这些区域的饱和度
        3. 对这些区域做轻微锐化，让边缘更清晰
        """

        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)

        # 绿色区域
        green_mask = (
            (h >= 35) & (h <= 90) &
            (s >= 20) &
            (v >= 50)
        )

        # 橙色 / 黄色 / 原木色区域
        yellow_orange_wood_mask = (
            (h >= 8) & (h <= 40) &
            (s >= 15) &
            (v >= 50)
        )

        # 合并 cube 可能的颜色区域
        cube_mask = (green_mask | yellow_orange_wood_mask).astype(np.uint8) * 255

        # 去掉小噪点
        kernel = np.ones((3, 3), np.uint8)
        cube_mask = cv2.morphologyEx(cube_mask, cv2.MORPH_OPEN, kernel)

        # 稍微扩张，覆盖 cube 边缘
        cube_mask = cv2.dilate(cube_mask, kernel, iterations=1)

        # 平滑边界，避免处理痕迹明显
        cube_mask = cv2.GaussianBlur(cube_mask, (9, 9), 0)

        mask_f = cube_mask.astype(np.float32) / 255.0
        mask_f = mask_f[..., None]

        # 增强 cube 的饱和度和亮度
        s_enhanced = np.clip(s * 2.3, 0, 255)
        v_enhanced = np.clip(v * 1.08, 0, 255)

        hsv_enhanced = cv2.merge([h, s_enhanced, v_enhanced]).astype(np.uint8)
        img_color_enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

        # 锐化边缘
        blur = cv2.GaussianBlur(img_color_enhanced, (0, 0), 1.0)
        img_sharp = cv2.addWeighted(img_color_enhanced, 1.7, blur, -0.7, 0)

        # 只在 cube 颜色区域融合增强结果
        out = img_bgr.astype(np.float32) * (1.0 - mask_f) + img_sharp.astype(np.float32) * mask_f

        return np.clip(out, 0, 255).astype(np.uint8)

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

    # def process_img(self, img_rgb):
    #     img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    #     # save_dir = f"output/task_reward_generation/{self.timestamp}"
    #     # os.makedirs(save_dir, exist_ok=True)
    #     # save_path = os.path.join(save_dir, f"scene_{round}.jpg")
    #     # # cv2.imwrite(save_path, img_bgr)
    #     # cv2.imwrite(save_path, img_rgb)

    #     ok, buffer = cv2.imencode(".jpg", img_bgr)
    #     if ok:
    #         b64jpg = base64.b64encode(buffer.tobytes()).decode("utf-8")
    #     else:
    #         raise RuntimeError("Failed to encode image to JPEG")
        
    #     # masked_img_rgb = self.mask_image(img_rgb)
    #     masked_img_bgr = self.mask_image(img_bgr)

    #     # return masked_img_rgb, b64jpg
    #     return masked_img_bgr, b64jpg
    
    def process_img(self, img_rgb):
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        # ok, buffer = cv2.imencode(".jpg", img_bgr)

        img_bgr_low_exp = self.reduce_exposure_bgr(
            img_bgr,
            alpha=0.55,  # 0.6 # 0.8
            beta=0,      # 0   # -30
        )
        ok, buffer = cv2.imencode(".jpg", img_bgr_low_exp)
        
        # # 1. 先增强 bowl，改善白色碗的结构
        # img_bgr_processed = self.enhance_bowl_visibility_bgr(img_bgr)
        # 2. 再增强 cube，提升颜色和边缘
        img_bgr_processed = self.enhance_cube_visibility_bgr(img_bgr_low_exp)  # img_bgr_processed
        ok, buffer = cv2.imencode(".jpg", img_bgr_processed)

        # save_dir = f"output/task_reward_generation/{self.timestamp}"
        # os.makedirs(save_dir, exist_ok=True)
        # save_path = os.path.join(save_dir, f"scene_{round}.jpg")
        # # cv2.imwrite(save_path, img_bgr)
        # cv2.imwrite(save_path, img_rgb)
        
        if ok:
            b64jpg = base64.b64encode(buffer.tobytes()).decode("utf-8")
        else:
            raise RuntimeError("Failed to encode image to JPEG")
        
        # masked_img_rgb = self.mask_image(img_rgb)
        # masked_img_bgr = self.mask_image(img_bgr_low_exp)
        masked_img_bgr = self.mask_image(img_bgr_processed)

        # return masked_img_rgb, b64jpg
        return masked_img_bgr, b64jpg

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def filter_gdino_boxes(self, boxes, logits, phrases, max_area_ratio=0.25):
        """
        去掉占图像面积太大的框，例如整张桌子的红色大框。
        boxes: GroundingDINO 输出，cxcywh，归一化坐标
        """
        keep = []

        for i, box in enumerate(boxes):
            cx, cy, w, h = box.tolist()
            area_ratio = w * h

            if area_ratio < max_area_ratio:
                keep.append(i)

        if len(keep) == 0:
            return boxes, logits, phrases

        boxes = boxes[keep]
        logits = logits[keep]
        phrases = [phrases[i] for i in keep]

        return boxes, logits, phrases

    def gdino(self, scene, before):
        # IMAGE_PATH = image_path # "weights/dog-3.jpeg"
        TEXT_PROMPT = self.selected_objects # "chair . person . dog ."
        BOX_THRESHOLD = 0.33 # 0.30 # 0.35
        TEXT_THRESHOLD = 0.22 # 0.22 # 0.25

        image_source, image = load_image(scene)  # IMAGE_PATH

        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )

        boxes, logits, phrases = self.filter_gdino_boxes(
            boxes, logits, phrases,
            max_area_ratio=0.25,
        )

        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

        save_dir = f"outputs/task_reward_generation/{self.timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        if before == True:
            save_path = os.path.join(save_dir, f"annotated_image_{self.round}_0_before.jpg")
        else:
            save_path = os.path.join(save_dir, f"annotated_image_{self.round}_1_after.jpg")
        cv2.imwrite(save_path, annotated_frame)

        scene_gdino = self.encode_image(save_path)

        return scene_gdino

    def task_generator(self, scene):
        tasks_text = "\n".join([f"- {task}" for task in self.candidate_tasks])

        # prompt_task = (
        #     "You are a one-arm robot. Based on the image, judge the spatial relationships between objects.\n"
        #     "From the task list below, randomly choose ONE feasible task for the current scene.\n"
        #     "Do not invent or modify tasks.\n\n"
        #     f"Task list:\n{tasks_text}\n\n"
        #     "Return exactly in this format:\n"
        #     "task: <selected task>\n"
        #     # "objects: object1 . object2 .\n"
        #     "objects: \"object1 . object2 .\"\n"
        # )

        prompt_task = (
            f"Task list:\n{tasks_text}\n\n"
            "You are a one-arm robot. Based on the image, judge the spatial relationships between objects.\n"
            "From the task list below, randomly choose ONE task that is feasible AND not already completed in the current scene.\n"
            "A task is already completed if its desired final spatial relationship is already true in the image.\n"
            "For example, if the cube is already inside the bowl, do NOT choose 'put the cube into the bowl'.\n"
            "Only choose a task whose goal state is currently false but can be achieved by the robot.\n"
            "Do not invent, rewrite, or modify tasks.\n\n"
            
            "Return exactly in this format:\n"
            "task: <selected task>\n"
            "objects: <object1> . <object2> .\n"
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
        self.task_generator(scene_gpt)
        
        self.round += 1
        self.scene_before = self.gdino(scene_dino, before=True)

        logger.info(f"Selected task to execute: {self.selected_task}")
        logger.info(f"Selected objects to detect: {self.selected_objects}")
        
        return self.selected_task, self.round
    
    def reward_generation(self, img_rgb, task_prompt = None):
        next_scene_dino, next_scene_gpt = self.process_img(img_rgb)
        self.scene_after = self.gdino(next_scene_dino, before=False)
        if task_prompt ==None:
            task = self.selected_task
        else:
            task = task_prompt
        reward = self.reward_generator(self.scene_before, self.scene_after, task)
        logger.info(f"Reward: {reward}")
        
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


