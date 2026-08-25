import os
import sys

import cv2
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
        save_images: bool = True,
    ):
        # Configuration
        self.max_timesteps = max_timesteps
        self.save_images = bool(save_images)
        self.current_scene = None
        self.next_scene = None

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

        # No GroundingDINO model is loaded in this variant.
        self.runtime_parameters = None

        # OpenAI client
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Fixed visual reference for the known drawer-closed and mug-on-tree states.
        self.scene_state_reference_path = (
            Path(__file__).resolve().parents[3]
            / "outputs"
            / "task_reward_generation"
            / "prompt.jpg"
        )
        if not self.scene_state_reference_path.is_file():
            raise FileNotFoundError(
                "Scene-state reference image not found at: "
                f"{self.scene_state_reference_path}"
            )
        self.scene_state_reference = base64.b64encode(
            self.scene_state_reference_path.read_bytes()
        ).decode("utf-8")

        # Logging

        self.round = 0
        self.scene_before = None
        self.scene_after = None
        self.selected_task = None
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
            "Open the drawer",
            "Close the drawer",

            # task 4
            "Hang the mug on the mug tree",
            "Take the mug off the mug tree",
        ]
        # Independent Beta(1, 1) success-rate estimates. Every task starts at
        # 0.5, and the pseudo success/failure counts keep estimates strictly
        # between 0 and 1 even after a long all-failure or all-success run.
        self.task_success_stats = {
            task: {"successes": 1.0, "failures": 1.0}
            for task in self.candidate_tasks
        }

    def get_task_success_rate(self, task: str) -> float:
        stats = self.task_success_stats[task]
        return stats["successes"] / (stats["successes"] + stats["failures"])

    def get_task_sampling_weight(self, task: str) -> float:
        """Prioritize tasks whose independently estimated success rate is low."""
        return 1.0 - self.get_task_success_rate(task)

    def update_task_success_rate(self, task: str, reward: int) -> float:
        if task not in self.task_success_stats:
            raise ValueError(f"Unknown task for success-rate update: {task!r}")
        if reward not in (0, 1):
            raise ValueError(f"Reward must be 0 or 1, got {reward!r}")
        key = "successes" if reward == 1 else "failures"
        self.task_success_stats[task][key] += 1.0
        return self.get_task_success_rate(task)

    def restore_task_success_stats(self, saved_stats: dict) -> None:
        """Restore validated per-task Beta counts from a training checkpoint."""
        if set(saved_stats) != set(self.candidate_tasks):
            raise ValueError(
                "Checkpoint task-success keys do not match candidate_tasks"
            )
        restored = {}
        for task in self.candidate_tasks:
            stats = saved_stats[task]
            successes = float(stats["successes"])
            failures = float(stats["failures"])
            if successes <= 0 or failures <= 0:
                raise ValueError(
                    f"Task success pseudo-counts must be positive for {task!r}"
                )
            restored[task] = {
                "successes": successes,
                "failures": failures,
            }
        # Mutate in place so any checkpointing thread holding this dictionary
        # keeps seeing the restored and subsequently updated values.
        self.task_success_stats.clear()
        self.task_success_stats.update(restored)

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

        # Configuration
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
        img_bgr_low_exp = self.reduce_exposure_bgr(
            img_bgr,
            # The ZED image is already fairly dark.  Reducing it to 55% made
            # the contact edge between two stacked cubes disappear.
            alpha=1.0,
            beta=0,
        )
        img_bgr_processed = self.enhance_cube_visibility_bgr(img_bgr_low_exp)
        masked_img_bgr = self.mask_image(img_bgr_processed)

        # GPT and the saved before/after artifacts must use the same masked image.
        ok, buffer = cv2.imencode(".jpg", masked_img_bgr)
        if ok:
            b64jpg = base64.b64encode(buffer.tobytes()).decode("utf-8")
        else:
            raise RuntimeError("Failed to encode masked image to JPEG")

        return masked_img_bgr, b64jpg

    def make_geometry_detail_image(self, encoded_scene):
        """Create a second full-scene view that emphasizes object boundaries."""
        encoded_bytes = base64.b64decode(encoded_scene)
        image = cv2.imdecode(
            np.frombuffer(encoded_bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if image is None:
            raise RuntimeError("Failed to decode scene for geometry enhancement")

        # CLAHE on luminance reveals the horizontal contact seam and cube side
        # faces without relying on a fixed cube location or changing hue.
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        luminance, channel_a, channel_b = cv2.split(lab)
        luminance = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(
            luminance
        )
        detail = cv2.cvtColor(
            cv2.merge([luminance, channel_a, channel_b]),
            cv2.COLOR_LAB2BGR,
        )
        blurred = cv2.GaussianBlur(detail, (0, 0), 1.0)
        detail = cv2.addWeighted(detail, 1.5, blurred, -0.5, 0)

        ok, buffer = cv2.imencode(".jpg", detail, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise RuntimeError("Failed to encode geometry-enhanced scene")
        return base64.b64encode(buffer.tobytes()).decode("utf-8")

    def has_two_separate_cube_top_faces(self, encoded_scene):
        """Return True when two independent cube top faces are clearly visible.

        This is a conservative veto for VLM stacking false positives.  A real
        stack normally exposes one dominant top face, whereas two cubes resting
        on the table expose two similarly sized, compact, saturated top faces.
        """
        image = cv2.imdecode(
            np.frombuffer(base64.b64decode(encoded_scene), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if image is None:
            logger.warning("Could not decode scene for cube-top-face validation")
            return False

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        green = (hue >= 35) & (hue <= 88)
        yellow = (hue >= 15) & (hue < 35)
        top_face_mask = (
            (green | yellow) & (saturation >= 80) & (value >= 70)
        ).astype(np.uint8) * 255
        top_face_mask = cv2.morphologyEx(
            top_face_mask,
            cv2.MORPH_OPEN,
            np.ones((3, 3), dtype=np.uint8),
        )

        image_h, image_w = image.shape[:2]
        image_area = image_h * image_w
        min_area = max(250, int(image_area * 0.00035))
        max_area = int(image_area * 0.01)
        min_width = max(12, int(image_w * 0.018))
        min_height = max(12, int(image_h * 0.025))
        _, _, stats, centroids = cv2.connectedComponentsWithStats(top_face_mask)

        candidates = []
        for component_index, (x, y, width, height, area) in enumerate(stats[1:], 1):
            if not (min_area <= area <= max_area):
                continue
            if width < min_width or height < min_height:
                continue
            aspect_ratio = width / float(height)
            fill_ratio = area / float(width * height)
            if not (0.55 <= aspect_ratio <= 1.8 and fill_ratio >= 0.45):
                continue
            center_x, center_y = centroids[component_index]
            # In this fixed overhead setup the cube/bowl manipulation zone is
            # the upper-left half of the masked tabletop. This rejects cyan
            # robot/cabinet highlights that happen to pass the HSV thresholds.
            if center_x >= image_w * 0.5 or center_y >= image_h * 0.55:
                continue
            candidates.append({
                "box": (int(x), int(y), int(width), int(height)),
                "area": int(area),
                "center": (float(center_x), float(center_y)),
            })

        for first_index, first in enumerate(candidates):
            x1, y1, w1, h1 = first["box"]
            for second in candidates[first_index + 1:]:
                x2, y2, w2, h2 = second["box"]
                area_ratio = first["area"] / float(second["area"])
                if not (0.4 <= area_ratio <= 2.5):
                    continue

                # Independent top faces must have a real background gap along
                # at least one image axis, rather than being overlapping layers.
                horizontal_gap = max(x1, x2) - min(x1 + w1, x2 + w2)
                vertical_gap = max(y1, y2) - min(y1 + h1, y2 + h2)
                if horizontal_gap < 3 and vertical_gap < 3:
                    continue

                center_distance = np.hypot(
                    first["center"][0] - second["center"][0],
                    first["center"][1] - second["center"][1],
                )
                mean_face_size = (w1 + h1 + w2 + h2) / 4.0
                if center_distance < 0.65 * mean_face_size:
                    continue

                logger.info(
                    "Cube vision veto found two separate top faces: "
                    f"{first['box']} and {second['box']}"
                )
                return True

        return False

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def capture_scene(self, encoded_scene, before):
        """Optionally save and always return a GPT-ready encoded scene."""
        if self.save_images:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            suffix = "0_before" if before else "1_after"
            save_path = self.output_dir / f"gpt_image_{self.round}_{suffix}.jpg"
            save_path.write_bytes(base64.b64decode(encoded_scene))
        return encoded_scene

    def task_generator(self, scene):
        geometry_detail_scene = self.make_geometry_detail_image(scene)
        prompt_task = (
            "You are given exactly three images.\n"
            "The first image is a fixed reference example showing the lower handled drawer "
            "in the CLOSED state (drawer_open=false) and the blue mug HANGING ON the mug tree "
            "(mug_on_tree=true). These two labels are ground-truth facts about the reference image. "
            "Use it only as a visual reference for recognizing those two states; ignore its cubes "
            "and bowl.\n"
            "The second and third images are two views of the same current scene to evaluate. "
            "Report every scene-state field from those current-scene images only. Do not copy "
            "the reference labels automatically; compare "
            "the current drawer and mug against the labeled reference.\n"
            "The scene contains exactly two cubes, one bowl, one drawer, one mug, and one mug tree.\n"

            "Determine the cube state in this strict order. FIRST decide cube_count_in_bowl from "
            "direct visual evidence. If cube_count_in_bowl=1, immediately set cubes_stacked=false "
            "and do not evaluate or infer stacking. ONLY if cube_count_in_bowl=0 may you evaluate "
            "whether the cubes are stacked.\n"
            "For cube_count_in_bowl, use ONLY image 2, the natural-color current scene. Do NOT "
            "use image 3 because its contrast enhancement can exaggerate reflections inside the "
            "bowl. Set cube_count_in_bowl=1 only when a distinct cube body, cube-shaped boundary, "
            "or cube-colored solid region is visibly inside or crossing the physical rim of the "
            "bowl. The white interior, cyan lighting, glare, reflections, shadows, and rim of an "
            "otherwise empty bowl are not a cube. If the bowl interior is empty or evidence of a "
            "cube inside it is ambiguous, set cube_count_in_bowl=0. The scene contains exactly two "
            "cubes, but this fact is context only: NEVER infer that a missing or occluded second cube "
            "must be in the bowl. Two stacked cubes can overlap and look like one tall object.\n"

            "Only after establishing cube_count_in_bowl=0, judge cubes_stacked conservatively. "
            "Here, stacked means vertical support "
            "in 3D: the upper cube's bottom face rests on the lower cube's top face. Position in the "
            "2D image is not physical height. Two cubes appearing one above the other in image pixels, "
            "touching side edges, being close together, or having a horizontal gap/seam does NOT mean "
            "they are stacked.\n"
            "Set cubes_stacked=false when two distinct cube top faces are both substantially visible, "
            "their centers are offset on the tabletop, their outer silhouettes are separately visible, "
            "or they merely sit adjacent to each other. In particular, two nearby green/yellow cubes "
            "arranged as an upper and lower object in the image while both top faces are visible are "
            "side-by-side on the table, not stacked.\n"
            "Set cubes_stacked=true only with clear 3D support and occlusion evidence: the upper cube "
            "substantially overlaps and occludes the lower cube's top face, the two tabletop-plane "
            "centers are nearly aligned, and only the lower cube's side or a small protruding portion "
            "is visible beneath the upper cube. A color boundary or contact seam alone is insufficient. "
            "The overhead camera can make a real stack resemble one tall combined object, but require "
            "the support/occlusion evidence above. If uncertain, set cubes_stacked=false. "
            "Use both current-scene views together for stacking judgments: they show the same instant, "
            "and image 3 only enhances luminance and edges to make contact boundaries easier to see.\n"

            "For drawer_open, evaluate ONLY the lower movable drawer that has the visible handle. "
            "Ignore the upper handleless panel/compartment completely; it is fixed and must always "
            "be treated as closed, so its appearance must never affect drawer_open.\n"
            "For the drawer judgment, use ONLY image 1 (the CLOSED reference) and image 2 (the natural-"
            "color current scene). Do NOT use image 3 because its contrast enhancement exaggerates "
            "normal seams and bright edges. Compare the position of the lower drawer FRONT PANEL, not "
            "the handle position, between images 1 and 2. The blue handle always protrudes even when "
            "the drawer is closed. White trim, rails, highlights, shadows, and narrow construction gaps "
            "that are also present in the CLOSED reference are normal closed-state features. They must "
            "never by themselves cause drawer_open=true.\n"
            "Set drawer_open=true only when image 2 provides clear evidence that the lower drawer front "
            "panel has moved substantially outward relative to its position in image 1, producing a "
            "new and substantially wider opening or clearly exposing the drawer interior. If the front "
            "panel position is approximately the same as the CLOSED reference, or if the evidence is "
            "ambiguous, set drawer_open=false.\n"

            "mug_on_tree is true only when the mug is currently hanging on the mug tree.\n"
            "mug_on_tree is false when the mug is not hanging on the mug tree.\n"

            "Do not evaluate tasks and do not return task names.\n"

            "Return only one valid JSON object with exactly these fields:\n"
            "{\"cube_count_in_bowl\": 0, "
            "\"cubes_stacked\": false, "
            "\"drawer_open\": false, "
            "\"mug_on_tree\": false}\n"
        )

        response = self.client.responses.create(
            model="gpt-5.6-terra",
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt_task},
                    {
                        "type": "input_text",
                        "text": (
                            "REFERENCE IMAGE (known labels): drawer_open=false, "
                            "mug_on_tree=true."
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": (
                            "data:image/jpeg;base64,"
                            f"{self.scene_state_reference}"
                        ),
                        "detail": "high",
                    },
                    {
                        "type": "input_text",
                        "text": (
                            "CURRENT SCENE IMAGE, natural-color view: classify this image."
                            " Use this view, together with the reference, for drawer_open."
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{scene}",
                        "detail": "high",
                    },
                    {
                        "type": "input_text",
                        "text": (
                            "CURRENT SCENE IMAGE, geometry-enhanced view of the exact same frame: "
                            "use it only to inspect cube boundaries and cube stacking. Ignore this "
                            "enhanced view when deciding cube_count_in_bowl or drawer_open, then "
                            "return the JSON state."
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{geometry_detail_scene}",
                        "detail": "high",
                    },
                ],
            }],
        )

        output = response.output_text.strip()
        json_start = output.find("{")
        json_end = output.rfind("}")
        if json_start == -1 or json_end == -1:
            raise RuntimeError(f"Scene-state response is not valid JSON: {output}")

        try:
            result = json.loads(output[json_start:json_end + 1])
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Scene-state response contains invalid JSON: {output}"
            ) from exc
        if not isinstance(result, dict):
            raise RuntimeError(f"Scene-state response must be a JSON object: {output}")
        expected_fields = {
            "cube_count_in_bowl",
            "cubes_stacked",
            "drawer_open",
            "mug_on_tree",
        }
        if set(result) != expected_fields:
            raise RuntimeError(
                f"Scene-state response must contain exactly {sorted(expected_fields)}: "
                f"{output}"
            )

        cube_count_in_bowl = result.get("cube_count_in_bowl")
        cubes_stacked = result.get("cubes_stacked")
        drawer_open = result.get("drawer_open")
        mug_on_tree = result.get("mug_on_tree")
        if type(cube_count_in_bowl) is not int or cube_count_in_bowl not in (0, 1):
            raise RuntimeError(
                f"cube_count_in_bowl must be integer 0 or 1: {output}"
            )
        if type(cubes_stacked) is not bool:
            raise RuntimeError(f"cubes_stacked must be a JSON boolean: {output}")
        if type(drawer_open) is not bool:
            raise RuntimeError(f"drawer_open must be a JSON boolean: {output}")
        if type(mug_on_tree) is not bool:
            raise RuntimeError(f"mug_on_tree must be a JSON boolean: {output}")

        # Bowl occupancy has priority over stacking. Enforce the hierarchy in
        # code too, even if the model returns an inconsistent combination.
        if cube_count_in_bowl == 1:
            if cubes_stacked:
                logger.warning(
                    "Ignoring cubes_stacked=true because "
                    "cube_count_in_bowl=1 takes priority"
                )
            cubes_stacked = False
        # Only inspect stacking after bowl occupancy has been ruled out. Do not
        # rely on the VLM alone for the common adjacent-cubes failure mode.
        elif cubes_stacked and self.has_two_separate_cube_top_faces(scene):
            logger.warning(
                "Overriding cubes_stacked=true: classical vision detected "
                "two separate cube top faces"
            )
            cubes_stacked = False

        evaluations = {
            "Put a cube into the bowl": {
                "feasible": cube_count_in_bowl == 0 and not cubes_stacked,
            },
            "Take the cube out of the bowl": {
                "feasible": cube_count_in_bowl == 1 and not cubes_stacked,
            },
            "Stack one cube on the other cube": {
                "feasible": cube_count_in_bowl == 0 and not cubes_stacked,
            },
            "Take the top cube off the other cube": {
                "feasible": cube_count_in_bowl == 0 and cubes_stacked,
            },
            "Open the drawer": {
                "feasible": not drawer_open,
            },
            "Close the drawer": {
                "feasible": drawer_open,
            },
            "Hang the mug on the mug tree": {
                "feasible": not mug_on_tree,
            },
            "Take the mug off the mug tree": {
                "feasible": mug_on_tree,
            },
        }
        feasible_tasks = [
            task
            for task in self.candidate_tasks
            if evaluations.get(task, {}).get("feasible") is True
        ]
        if not feasible_tasks:
            raise RuntimeError(
                "No candidate task is feasible for scene state "
                f"cube_count_in_bowl={cube_count_in_bowl}, "
                f"cubes_stacked={cubes_stacked}, "
                f"drawer_open={drawer_open}, "
                f"mug_on_tree={mug_on_tree}"
            )

        weights = [
            self.get_task_sampling_weight(task)
            for task in feasible_tasks
        ]
        feasible_weight_total = sum(weights)
        if feasible_weight_total <= 0:
            raise RuntimeError(
                f"Feasible task weights must sum to a positive value: {weights}"
            )
        self.selected_task = random.choices(
            feasible_tasks, weights=weights, k=1
        )[0]
        logger.info(
            "Scene state: "
            f"cube_count_in_bowl={cube_count_in_bowl}, "
            f"cubes_stacked={cubes_stacked}, "
            f"drawer_open={drawer_open}, "
            f"mug_on_tree={mug_on_tree}"
        )
        formatted_tasks = ", ".join(
            f"{task}: {weight / feasible_weight_total:.6f}"
            for task, weight in zip(feasible_tasks, weights)
        )
        logger.info(
            "Feasible tasks (conditional probabilities): "
            f"{formatted_tasks}"
        )
        formatted_success_rates = ", ".join(
            f"{task}: {self.get_task_success_rate(task):.6f}"
            for task in self.candidate_tasks
        )
        logger.info(f"Task success rates: {formatted_success_rates}")

    def reward_generator(self, current_scene, next_scene, current_task):
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
            "4. Carefully distinguish objects with similar colors or shapes using their "
            "appearance and spatial context.\n"
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
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{current_scene}", "detail": "high"},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{next_scene}", "detail": "high"},
                ],
            }],
        )

        reward = response.output_text.strip()
        if reward not in {"0", "1"}:
            raise RuntimeError(
                f"Reward response must be exactly 0 or 1, got {reward!r}"
            )

        return reward

    # task_0-(reward_0-task_1)-(reward_1-task_2)-...
    def start_task(self, img_rgb):
        """Capture the before-scene for an externally supplied evaluation task."""
        _, scene_gpt = self.process_img(img_rgb)
        self.round += 1
        self.scene_before = self.capture_scene(scene_gpt, before=True)

    def task_generation(self, img_rgb):  # shape: (1080, 1920, 3) dtype: uint8
        _, scene_gpt = self.process_img(img_rgb)
        last_error = None
        for attempt in range(1, 4):
            try:
                self.task_generator(scene_gpt)
                break
            except RuntimeError as exc:
                last_error = exc
                logger.warning(
                    f"Scene-state recognition attempt {attempt}/3 failed: {exc}"
                )
        else:
            weights = [
                self.get_task_sampling_weight(task)
                for task in self.candidate_tasks
            ]
            self.selected_task = random.choices(
                self.candidate_tasks, weights=weights, k=1
            )[0]
            logger.error(
                "GPT could not identify a valid scene state after 3 attempts; "
                f"using probability-weighted fallback task "
                f"'{self.selected_task}'. Last error: {last_error}"
            )
        
        self.round += 1
        self.scene_before = self.capture_scene(scene_gpt, before=True)

        logger.info(f"Selected task to execute: {self.selected_task}")
        
        return self.selected_task, self.round
    
    def reward_generation(self, img_rgb, task_prompt=None, log_separator=True):
        _, next_scene_gpt = self.process_img(img_rgb)
        self.scene_after = self.capture_scene(next_scene_gpt, before=False)
        if task_prompt ==None:
            task = self.selected_task
        else:
            task = task_prompt
        reward = self.reward_generator(self.scene_before, self.scene_after, task)
        success_rate = self.update_task_success_rate(task, int(reward))
        logger.info(f"Reward: {reward}")
        logger.info(
            f"Updated task success rate: {task}: {success_rate:.6f}"
        )
        if log_separator:
            logger.info("-------------------------------------------------------------------")
        
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
