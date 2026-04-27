import contextlib
import signal
import threading

import numpy as np
from PIL import Image
from openpi_client import image_tools
import cv2

@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Only install SIGINT handler in main thread."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

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

def save_np_image(img: np.ndarray, path: str):
    assert img.dtype == np.uint8
    assert img.ndim == 3 and img.shape[2] == 3
    Image.fromarray(img).save(path)

# def center_crop_square(img_hwc: np.ndarray) -> np.ndarray:
#     """Center-crop HWC image to square."""
#     h, w = img_hwc.shape[:2]
#     side = min(h, w)
#     y0 = (h - side) // 2
#     x0 = (w - side) // 2
#     return img_hwc[y0:y0 + side, x0:x0 + side]

def prepare_image_256(img, size=(256, 256)):  # TODO: 一步到位
    """中心裁剪成正方形，再缩放到指定大小 (默认 256x256)，输出 RGB uint8。"""
    if img is None:
        raise ValueError("prepare_image_256 got None")
    
    h, w = img.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    crop = img[y0:y0 + side, x0:x0 + side]
    out = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
    return out.astype(np.uint8, copy=False)

def to_hwc(img):
    if img is None:
        return None

    img = np.asarray(img)

    # (1, 3, H, W) -> (3, H, W)
    if img.ndim == 4:
        if img.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, got shape={img.shape}")
        img = img[0]

    # CHW(3, H, W) -> HWC(H, W, 3)
    if img.ndim == 3 and img.shape[0] in [1, 3]:
        img = np.transpose(img, (1, 2, 0))

    if img.ndim == 2:
        img = img[..., None]

    if img.ndim != 3:
        raise ValueError(f"Invalid image ndim={img.ndim}, shape={img.shape}")

    return img

def process_policy_images(obs_left, obs_right, obs_wrist):
    obs_left = prepare_image_256(to_hwc(obs_left))
    obs_right = prepare_image_256(to_hwc(obs_right))
    obs_wrist = prepare_image_256(to_hwc(obs_wrist))  # padding

    left_resized = image_tools.resize_with_pad(obs_left, 224, 224)
    right_resized = image_tools.resize_with_pad(obs_right, 224, 224)
    wrist_resized = image_tools.resize_with_pad(obs_wrist, 224, 224)

    return left_resized, right_resized, wrist_resized

def _extract_observation(obs_dict, *, save_to_disk=False):
    image_observations = obs_dict["image"]
    left_image, right_image, wrist_image = None, None, None
    for key in image_observations:
        # Note the "left" below refers to the left camera in the stereo pair.
        # The model is only trained on left stereo cams, so we only feed those.
        if "left_cam" in key:
            left_image = image_observations[key]
        elif "right_cam" in key:
            right_image = image_observations[key]
        elif "wrist_cam" in key:
            wrist_image = image_observations[key]

    # Drop the alpha dimension
    left_image = left_image[..., :3]
    right_image = right_image[..., :3]
    wrist_image = wrist_image[..., :3]

    # Convert to RGB
    left_image = left_image[..., ::-1]
    right_image = right_image[..., ::-1]
    wrist_image = wrist_image[..., ::-1]

    robot_state = obs_dict["robot_state"]
    cartesian_position = np.array(robot_state["cartesian_position"])
    joint_position = np.array(robot_state["joint_positions"])
    gripper_position = np.array([robot_state["gripper_position"]])

    if save_to_disk:
        combined_image = np.concatenate([left_image, wrist_image, right_image], axis=1)
        Image.fromarray(combined_image).save("robot_camera_views.png")

    return {
        "left_image": left_image,
        "right_image": right_image,
        "wrist_image": wrist_image,
        "cartesian_position": cartesian_position,
        "joint_position": joint_position,
        "gripper_position": gripper_position,
    }


# def prepare_image_256(img, size=(256, 256)):  # TODO: 一步到位
#     """中心裁剪成正方形，再缩放到指定大小 (默认 256x256)，输出 RGB uint8。"""
#     if img is None:
#         print("❌ img is None")
#         return None
#     h, w = img.shape[:2]
#     side = min(h, w)
#     y0 = (h - side) // 2
#     x0 = (w - side) // 2
#     crop = img[y0:y0 + side, x0:x0 + side]
#     # out = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
#     return out.astype(np.uint8, copy=False)

# def process_policy_images(
#     obs_left,
#     obs_right,
#     obs_wrist,
#     base_size: int = None,
#     residual_size: int = None,
# ) -> dict:
#     """
#     Unified image preprocessing for both:
#     - base policy / pi05 input: 224x224 (with pad)
#     - residual actor / learner input: residual_size x residual_size
#     """
    
#     if base_size:
#         obs_left = prepare_image_256(to_hwc(obs_left))
#         obs_right = prepare_image_256(to_hwc(obs_right))
#         obs_wrist = prepare_image_256(to_hwc(obs_wrist))

#         left_resized = image_tools.resize_with_pad(obs_left, base_size, base_size)
#         right_resized = image_tools.resize_with_pad(obs_right, base_size, base_size)
#         wrist_resized = image_tools.resize_with_pad(obs_wrist, base_size, base_size)

#     if residual_size:
#         left_resized = resize_hwc(left, residual_size)
#         right_resized = resize_hwc(right, residual_size)
#         wrist_resized = resize_hwc(wrist, residual_size)

#     return left_resized, right_resized, wrist_resized




