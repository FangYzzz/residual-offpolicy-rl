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
from groundingdino.util.inference import Model, annotate
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
        self.output_root = Path("outputs/task_reward_generation") / self.timestamp
        self.output_dir = self.output_root
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.queue: "Queue[tuple[list[str], list[str]]]" = Queue()

        # Hardware / model setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Model(
            model_config_path="/home/yuan/self_vla/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            model_checkpoint_path="/home/yuan/self_vla/GroundingDINO/weights/groundingdino_swint_ogc.pth",
            device=self.device,
        )
        self.runtime_parameters = None

        # OpenAI client
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Logging

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
            # "put the cube into the bowl",
            # "put the cube outside the bowl",
            ### dataset ###
            # task 1
            "Put a cube into the bowl",
            "Take the cube out of the bowl",

            # task 2
            "Stack one cube on the other cube",
            "Take the top cube off the other cube",

            # task 3
            # "Open the drawer",
            # "Close the drawer",

            # task 4
            # "Hang the mug on the mug tree",
            # "Take the mug off the mug tree",
        ]
        # Relative generation probabilities. Infeasible tasks are removed first,
        # then these weights are normalized over the remaining feasible tasks.
        self.task_probabilities = {
            "Put a cube into the bowl": 0.33,
            "Take the cube out of the bowl": 0.33,
            "Stack one cube on the other cube": 0.34,
        }

    def setup_logger(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.output_dir / "log.txt"

        logger.remove()
        logger.add(str(log_path), format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO")
        logger.add(sys.stdout, colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
        # logger.add(lambda msg: print(msg, end=""), format="{message}")

    def set_output_cycle(self, global_step: int) -> Path:
        """Route evaluation and following training artifacts to one step folder."""
        self.output_dir = self.output_root / f"step_{int(global_step)}"
        self.setup_logger()
        return self.output_dir

    def set_output_warmup(self) -> Path:
        """Route initial online replay-buffer collection artifacts to warmup/."""
        self.output_dir = self.output_root / "warmup"
        self.setup_logger()
        logger.info("---------------- warmup: collecting online replay buffer ----------------")
        return self.output_dir

    def restore_output_root(self, output_root: str | Path) -> None:
        """Reuse the artifact root stored in a resumed training checkpoint."""
        self.output_root = Path(output_root)
        self.output_dir = self.output_root

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
            [610, 520],   # 左1/2内
            [350, 510],   # 左1/2外
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

    def filter_gdino_detections(self, detections, image_shape, max_area_ratio=0.25):
        """
        去掉占图像面积太大的框，例如整张桌子的红色大框。
        detections: GroundingDINO new API 输出，xyxy 绝对像素坐标
        """
        image_h, image_w = image_shape[:2]
        image_area = image_h * image_w
        boxes = detections.xyxy
        area_ratios = (
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        ) / image_area
        keep = area_ratios < max_area_ratio

        # Preserve the previous behavior: if every box is too large, retain all.
        return detections[keep] if np.any(keep) else detections

    def gdino(self, scene, before):
        # if isinstance(self.selected_objects, str):
        #     classes = [
        #         class_name.strip()
        #         for class_name in self.selected_objects.split(".")
        #         if class_name.strip()
        #     ]
        # else:
        #     classes = list(self.selected_objects)

        classes = ["cube", "bowl", "mug", "mug tree", "drawer"]

        BOX_THRESHOLD = 0.25 #0.33 # 0.30 # 0.35  
        TEXT_THRESHOLD = 0.18 # 0.22 # 0.25

        detections = self.model.predict_with_classes(
            image=scene,
            classes=classes,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )

        detections = self.filter_gdino_detections(
            detections,
            scene.shape,
            max_area_ratio=0.25,
        )

        # Keep using the existing annotation helper, which expects normalized
        # cxcywh boxes and phrase labels rather than supervision.Detections.
        image_h, image_w = scene.shape[:2]
        boxes_xyxy = torch.as_tensor(detections.xyxy, dtype=torch.float32)
        boxes = torch.empty_like(boxes_xyxy)
        boxes[:, 0] = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / (2 * image_w)
        boxes[:, 1] = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / (2 * image_h)
        boxes[:, 2] = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) / image_w
        boxes[:, 3] = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]) / image_h
        logits = torch.as_tensor(detections.confidence, dtype=torch.float32)
        phrases = [
            classes[class_id] if class_id is not None else "unknown"
            for class_id in detections.class_id
        ]
        # scene is BGR for the new API; annotate expects an RGB source image.
        image_source = cv2.cvtColor(scene, cv2.COLOR_BGR2RGB)
        annotated_frame = annotate(
            image_source=image_source,
            boxes=boxes,
            logits=logits,
            phrases=phrases,
        )

        save_dir = self.output_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        if before == True:
            save_path = save_dir / f"annotated_image_{self.round}_0_before.jpg"
        else:
            save_path = save_dir / f"annotated_image_{self.round}_1_after.jpg"
        cv2.imwrite(str(save_path), annotated_frame)

        scene_gdino = self.encode_image(str(save_path))

        return scene_gdino

    def task_generator(self, scene):
        tasks_text = "\n".join([f"- {task}" for task in self.candidate_tasks])
        prompt_task = (
            f"Task list:\n{tasks_text}\n\n"

            "You are a one-arm robot. Based on the image, determine the current spatial "
            "relationships and states of the objects.\n"
            "The scene contains exactly two cubes, one bowl, one drawer, one mug, and one mug tree.\n\n"

            "Evaluate EVERY task in the task list. Mark a task feasible only when its goal "
            "state is not already satisfied and all task-specific preconditions are met.\n"
            "Do not choose a task yourself; the caller will sample from the feasible tasks.\n"
            "The following task-specific rules are hard constraints and must never be violated.\n\n"

            "Task-specific eligibility rules:\n\n"

            "1. Cube and bowl tasks\n"
            "- 'Put a cube into the bowl' may be selected only when the bowl contains no cube.\n"
            "- 'Take the cube out of the bowl' may be selected only when the bowl contains exactly one cube.\n"
            "- The bowl may contain at most one cube. Never select a task that would place "
            "a second cube into the bowl.\n\n"

            "2. Cube stacking tasks\n"
            "- 'Stack one cube on the other cube' may be selected only when the bowl contains no cube, "
            "both cubes are outside the bowl, and the cubes are not already stacked.\n"
            "- 'Take the top cube off the other cube' may be selected only when one cube is "
            "currently stacked on top of the other cube.\n\n"

            # "3. Drawer tasks\n"
            # "- 'Open the drawer' may be selected only when the drawer is currently closed.\n"
            # "- 'Close the drawer' may be selected only when the drawer is currently open.\n\n"

            # "4. Mug and mug-tree tasks\n"
            # "- 'Hang the mug on the mug tree' may be selected only when the mug is not "
            # "currently hanging on the mug tree.\n"
            # "- 'Take the mug off the mug tree' may be selected only when the mug is currently "
            # "hanging on the mug tree.\n\n"

            "Do not invent, rewrite, or modify any task from the task list.\n"
            "Return only the required output, with no explanation or additional text.\n\n"

            "Return exactly one valid JSON object in this format:\n"
            "{\n"
            "  \"tasks\": [\n"
            "    {\"task\": \"<task copied verbatim from the task list>\", "
            "\"feasible\": true, \"objects\": \"<object1> . <object2> .\"}\n"
            "  ]\n"
            "}\n"
            "Include exactly one entry for every task in the task list. Use JSON booleans "
            "true and false. For an infeasible task, objects may be an empty string.\n"
        )

        # prompt_task = (
        #     f"Task list:\n{tasks_text}\n\n"
        #     "You are a one-arm robot. Based on the image, judge the spatial relationships between objects.\n"
        #     "From the task list below, randomly choose ONE task that is feasible AND not already completed in the current scene.\n"
        #     "A task is already completed if its desired final spatial relationship is already true in the image.\n"
        #     "For example, if a cube is already inside the cube bowl, do NOT choose 'Put a cube into the bowl'.\n"
        #     "Instead, you may choose 'Take the cube out of the bowl' if it is feasible.\n"
        #     "Do not choose 'Stack one cube on the other cube' unless both cubes are outside the bowl.\n"
        #     "Only choose a task whose goal state is currently false but can be achieved by the robot.\n"
        #     "Do not invent, rewrite, or modify tasks.\n\n"
            
        #     "Return exactly in this format:\n"
        #     "task: <selected task>\n"
        #     "objects: <object1> . <object2> .\n"
        # )

        response = self.client.responses.create(
            model="gpt-5.6-terra",
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt_task},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{scene}", "detail": "high"},
                ],
            }],
        )

        output = response.output_text.strip()
        json_start = output.find("{")
        json_end = output.rfind("}")
        if json_start == -1 or json_end == -1:
            raise RuntimeError(f"Task feasibility response is not valid JSON: {output}")

        try:
            result = json.loads(output[json_start:json_end + 1])
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Task feasibility response contains invalid JSON: {output}"
            ) from exc
        if not isinstance(result, dict) or not isinstance(result.get("tasks"), list):
            raise RuntimeError(
                f"Task feasibility JSON must contain a 'tasks' list: {output}"
            )
        evaluations = {
            item["task"]: item
            for item in result.get("tasks", [])
            if isinstance(item, dict) and item.get("task") in self.candidate_tasks
        }
        feasible_tasks = [
            task for task in self.candidate_tasks
            if evaluations.get(task, {}).get("feasible") is True
        ]
        if not feasible_tasks:
            raise RuntimeError("No task satisfies the current scene constraints.")

        weights = [self.task_probabilities.get(task, 1.0) for task in feasible_tasks]
        selected_task = random.choices(feasible_tasks, weights=weights, k=1)[0]
        selected_objects = evaluations[selected_task].get("objects", "").strip()
        if not selected_objects:
            raise RuntimeError(f"No objects returned for selected task: {selected_task}")

        self.selected_task = selected_task
        self.selected_objects = selected_objects
        logger.info(f"Feasible tasks: {feasible_tasks}; weights: {weights}")

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
            model="gpt-5.6-terra",
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt_reward},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{current_scene_gdino}", "detail": "high"},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{next_scene_gdino}", "detail": "high"},
                ],
            }],
        )

        reward = response.output_text

        return reward

    # task_0-(reward_0-task_1)-(reward_1-task_2)-...
    def start_task(self, img_rgb):
        """Capture the before-scene for an externally supplied evaluation task."""
        scene_dino, _ = self.process_img(img_rgb)
        self.round += 1
        self.scene_before = self.gdino(scene_dino, before=True)

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
