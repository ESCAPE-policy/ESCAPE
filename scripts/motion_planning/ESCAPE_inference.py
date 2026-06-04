import sys
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SOURCE_ROOT = os.path.join(REPO_ROOT, "source", "ESCAPE")
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

# geometrout uses numba cache=True during import. Keep that cache in /tmp so
# inference does not depend on write access or locator metadata in site-packages.
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/escape_numba_cache")

import argparse

os.environ["ESCAPE_ENABLE_RADAR"] = "true"

OSS_INFERENCE_DEFAULTS = {
    "task": "Isaac-Franka-HDF5-USD-all-v0",
    "target_hdf5_path": os.path.join(
        REPO_ROOT,
        "assets/scenes/targets/escape_public_targets.hdf5",
    ),
    "num_envs": 1,
    "num_cycles": 1,
    "num_inference_steps": 10,
    "goal_ghost_opacity": 0.3,
    "goal_ghost_transparency": 0.8,
    "goal_ghost_mdl_path": None,
    "goal_ghost_mdl_material": None,
    "goal_ghost_color": (1.0, 1.0, 1.0),
    "goal_ghost_emissive_strength": 0.0,
    "collision_margin": 0.005,
    "tracking_error_threshold": 0.15,
    "velocity_threshold": 1.0,
    "sustained_steps": 3,
}

parser = argparse.ArgumentParser(description="ESCAPE radar inference")
parser.add_argument("--checkpoint_path", type=str, required=True)
parser.add_argument("--scene_name", type=str, required=True, choices=("scene_1", "scene_2", "scene_3"))
parser.add_argument("--demo_id", type=int, default=0, choices=(0,))
parser.add_argument("--headless", action="store_true")

args_cli = parser.parse_args()
for default_name, default_value in OSS_INFERENCE_DEFAULTS.items():
    setattr(args_cli, default_name, default_value)
args_cli.device = "cuda:0"

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(
    {
        "headless": args_cli.headless,
        "device": args_cli.device,
    }
)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from collections import deque
from pxr import Usd, UsdShade, Sdf
from termcolor import cprint
import h5py
import omni.usd
import omni.timeline

import ESCAPE.tasks.manager_based.usd_scene_all.config.franka
from isaaclab_tasks.utils import parse_env_cfg
from ESCAPE.motion_planners.maniflow.maniflow_planner import ManiFlowPlanner
from ESCAPE.motion_planners.curobo.curobo_planner import CuroboPlanner
from ESCAPE.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg
from ESCAPE.utils.ghost_robot_utils import show_goal_as_ghost
from ESCAPE.utils.collision_checker import CollisionChecker
from ESCAPE.utils.usd_geometry_utils import (
    build_robot_pointcloud_cache_from_usd,
    collect_robot_pointcloud_from_fk,
    collect_robot_pointcloud_from_usd,
)
from ESCAPE.utils.radar_pointcloud_utils import (
    discover_lidars,
)
from ESCAPE.utils.radar_pointcloud_headless_utils import (
    HeadlessRobotSelfFilter,
    HeadlessRaycastRadarPointCloudCollector,
)
from curobo.types.state import JointState

RobotSelfFilter = HeadlessRobotSelfFilter


def load_debug_draw_interface():
    """Load Isaac debug_draw only when viewport radar visualization is enabled."""
    try:
        from isaacsim.util.debug_draw import _debug_draw
    except ModuleNotFoundError:
        from omni.isaac.debug_draw import _debug_draw
    return _debug_draw.acquire_debug_draw_interface()


def disable_native_lidar_visuals(stage, env_prefix="/World/envs/env_0") -> int:
    """Hide USD/native LiDAR point and line drawing before custom radar visualization starts."""
    disabled = 0
    attr_names = (
        "drawPoints",
        "drawLines",
        "rangeSensor:drawPoints",
        "rangeSensor:drawLines",
    )
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if env_prefix not in path:
            continue

        name = prim.GetName()
        type_name = prim.GetTypeName()
        if not name.startswith("Lidar") and "Lidar" not in type_name and "RangeSensor" not in type_name:
            continue

        changed = False
        for attr_name in attr_names:
            attr = prim.GetAttribute(attr_name)
            if not attr.IsValid():
                continue
            attr.Set(False)
            changed = True
        if changed:
            disabled += 1

    return disabled


# Constants
TOTAL_ROBOT_POINTS = 1024
TOTAL_OBSTACLE_POINTS = 2048
RADAR_MERGE_FRAMES = 10
RADAR_FRAME_STRIDE = 3
RAYCAST_RADAR_POINT_SIZE = 8.0
RAYCAST_RADAR_VISUAL_STRIDE = 1
OBSTACLE_COLOR = np.array([1.0, 0.0, 0.0], dtype=np.float32)
ROBOT_COLOR = np.array([0.0, 0.0, 1.0], dtype=np.float32)
GOAL_THRESHOLD_POSITION_M = 0.01
GOAL_THRESHOLD_ORIENTATION_DEG = 15.0
MAX_TASK_STEPS_NO_COLLISION = 400
MAX_TASK_STEPS_WITH_COLLISION = 300
MAX_TASK_STEPS = MAX_TASK_STEPS_NO_COLLISION
N_ACTION_STEPS = 15
TEMPERATURE=0.2
SELF_FILTER_THRESHOLD = 0.01
SELF_FILTER_USE_YAML = True
SELF_FILTER_SPHERES_YAML = os.path.join(REPO_ROOT, "assets/franka_radar_mesh_for_point_cloud_filter.yml")
ROBOT_PC_EXCLUDED_BODIES = []
ROBOT_PC_BODY_WEIGHTS = {
    "panda_link0": 0.2,
    "panda_link1": 0.2,
    "panda_hand": 4.0,
    "panda_leftfinger": 6.0,
    "panda_rightfinger": 6.0,
}
ROBOT_PC_BASE_FILTER_ENABLED = False
ROBOT_PC_BASE_FILTER_BODY = "panda_link0"
ROBOT_PC_BASE_FILTER_RADIUS = 0.18
ROBOT_PC_BASE_FILTER_HEIGHT = 0.20
EE_BODY_NAME = "panda_hand"
FRANKA_ARM_JOINT_LIMITS = np.array([
    (-2.8973, 2.8973),
    (-1.7628, 1.7628),
    (-2.8973, 2.8973),
    (-3.0718, -0.0698),
    (-2.8973, 2.8973),
    (-0.0175, 3.7525),
    (-2.8973, 2.8973),
], dtype=np.float32)
JOINT_LIMIT_DEBUG_THRESHOLD = 0.1
JOINT_LIMIT_STILL_VELOCITY_THRESHOLD = 0.1


class State(Enum):
    EXECUTING = "executing"
    DONE = "done"


@dataclass
class Task:
    env_id: int
    state: State
    planner: Optional[any] = None
    # ACT Temporal Ensembling: action buffer for each timestep
    action_buffer: Dict[int, List[torch.Tensor]] = field(default_factory=dict)
    # Store the predicted action chunk from last inference
    last_action_chunk: Optional[torch.Tensor] = None
    total_steps: int = 0
    success: bool = False
    collision_detected: bool = False
    collision_steps: int = 0
    replan_count: int = 0
    obs_queue: Optional[deque] = None
    radar_history: Optional[deque] = None
    max_tracking_error: float = 0.0


def get_task_max_steps(task: Task) -> int:
    return MAX_TASK_STEPS_WITH_COLLISION if task.collision_detected else MAX_TASK_STEPS_NO_COLLISION


def quaternion_angular_error_deg(quat_xyzw_a: np.ndarray, quat_xyzw_b: np.ndarray) -> float:
    """Return the smallest angular distance between two XYZW quaternions in degrees."""
    quat_a = np.asarray(quat_xyzw_a, dtype=np.float64)
    quat_b = np.asarray(quat_xyzw_b, dtype=np.float64)
    quat_a /= max(np.linalg.norm(quat_a), 1e-12)
    quat_b /= max(np.linalg.norm(quat_b), 1e-12)
    dot = float(np.clip(np.abs(np.dot(quat_a, quat_b)), -1.0, 1.0))
    angle_rad = 2.0 * np.arccos(dot)
    return float(np.degrees(angle_rad))


def get_ee_pose_from_joints(joint_angles: np.ndarray, planner) -> tuple[np.ndarray, np.ndarray]:
    """Compute end-effector position and XYZW quaternion from 7-DoF joint angles."""
    joint_tensor = torch.tensor(
        joint_angles,
        dtype=torch.float32,
        device=planner.tensor_args.device,
    )
    joint_state = JointState(
        position=joint_tensor.unsqueeze(0),
        velocity=torch.zeros_like(joint_tensor).unsqueeze(0),
        acceleration=torch.zeros_like(joint_tensor).unsqueeze(0),
        jerk=torch.zeros_like(joint_tensor).unsqueeze(0),
        joint_names=None,
        tensor_args=planner.tensor_args,
    )

    ee_pose = planner.get_ee_pose(joint_state)
    position = ee_pose.position.squeeze().cpu().numpy()
    quat_wxyz = ee_pose.quaternion.squeeze().cpu().numpy()
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    return position, quat_xyzw


def compute_ee_goal_errors(
    current_joints: np.ndarray,
    target_ee_pose: np.ndarray,
    planner,
) -> tuple[float, float]:
    """Compute end-effector position and orientation error against target EE pose."""
    current_ee_position, current_ee_quaternion_xyzw = get_ee_pose_from_joints(
        np.asarray(current_joints[:7], dtype=np.float32),
        planner,
    )
    target_ee_pose = np.asarray(target_ee_pose, dtype=np.float32)
    target_ee_position = target_ee_pose[:3]
    target_ee_quaternion_xyzw = target_ee_pose[3:7]

    position_error_m = float(np.linalg.norm(current_ee_position - target_ee_position))
    orientation_error_deg = quaternion_angular_error_deg(
        current_ee_quaternion_xyzw,
        target_ee_quaternion_xyzw,
    )
    return position_error_m, orientation_error_deg


# ==================== Point Cloud Collection ====================

def add_color_to_pointcloud(pc: np.ndarray, color: np.ndarray) -> np.ndarray:
    """Add RGB color to an XYZ point cloud."""
    if pc is None or pc.size == 0:
        return np.zeros((0, 6), dtype=np.float32)

    xyz = np.asarray(pc, dtype=np.float32).reshape(-1, 3)
    rgb = np.tile(np.asarray(color, dtype=np.float32), (xyz.shape[0], 1))
    return np.concatenate([xyz, rgb], axis=1).astype(np.float32)


def add_obstacle_color_with_depth_channel(pc_xyzd: np.ndarray, base_color: np.ndarray) -> np.ndarray:
    """Add RGB to obstacle XYZD and write processed depth into the G channel."""
    if pc_xyzd is None or pc_xyzd.size == 0:
        return np.zeros((0, 6), dtype=np.float32)

    arr = np.asarray(pc_xyzd, dtype=np.float32).reshape(-1, 4)
    xyz = arr[:, :3]
    depth_intensity = np.clip(arr[:, 3], 0.0, 1.0).astype(np.float32)

    colors = np.tile(np.asarray(base_color, dtype=np.float32), (arr.shape[0], 1))
    colors[:, 1] = depth_intensity
    return np.concatenate([xyz, colors], axis=1).astype(np.float32)


def get_radar_merge_layout(
    window_size: int,
    far_stride: int,
) -> tuple[int, int, int]:
    """Match dataset generation: recent dense frames plus older sparse frames."""
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if far_stride <= 0:
        raise ValueError(f"far_stride must be positive, got {far_stride}")

    recent_count = (window_size + 1) // 2
    older_count = window_size - recent_count
    required_history = recent_count + older_count * far_stride
    return recent_count, older_count, required_history


def merge_recent_pointcloud_frames_with_depth(
    frame_history: deque,
    window_size: int,
    frame_stride: int,
) -> np.ndarray:
    """Match dataset generation by using recent dense and older sparse history."""
    if not frame_history:
        return np.zeros((0, 4), dtype=np.float32)

    valid_history = []
    for pc in frame_history:
        if pc is None:
            continue
        arr = np.asarray(pc, dtype=np.float32)
        if arr.size == 0:
            continue
        valid_history.append(arr.reshape(-1, 4))

    if not valid_history:
        return np.zeros((0, 4), dtype=np.float32)

    latest_idx = len(valid_history) - 1
    latest_frame = valid_history[latest_idx]
    recent_count, older_count, _ = get_radar_merge_layout(window_size, far_stride=frame_stride)
    selected_rev = []

    for i in range(recent_count):
        hist_idx = latest_idx - i
        if hist_idx >= 0:
            selected_rev.append(valid_history[hist_idx])
        else:
            selected_rev.append(latest_frame.copy())

    older_latest_idx = latest_idx - recent_count
    adaptive_far_stride = min(
        frame_stride,
        max(1, older_latest_idx // max(older_count, 1)) if older_latest_idx >= 0 else 1,
    )

    for i in range(older_count):
        hist_idx = older_latest_idx - i * adaptive_far_stride
        if hist_idx >= 0:
            selected_rev.append(valid_history[hist_idx])
        else:
            selected_rev.append(latest_frame.copy())

    selected = list(reversed(selected_rev))
    return np.concatenate(selected, axis=0).astype(np.float32)


def filter_points_and_depth_by_return_value(
    points: np.ndarray,
    return_values: np.ndarray,
    fallback_depth_values: Optional[np.ndarray] = None,
    invalid_return_value: float = 4.0,
    atol: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter points and aligned depth by per-point lidar return value."""
    if points is None:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if pts.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    if return_values is not None:
        depth = np.asarray(return_values, dtype=np.float32).reshape(-1)
    elif fallback_depth_values is not None:
        depth = np.asarray(fallback_depth_values, dtype=np.float32).reshape(-1)
    else:
        depth = np.zeros((pts.shape[0],), dtype=np.float32)

    if depth.shape[0] != pts.shape[0]:
        min_n = min(depth.shape[0], pts.shape[0])
        if min_n <= 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        pts = pts[:min_n]
        depth = depth[:min_n]

    valid_mask = np.isfinite(depth)
    if return_values is not None:
        valid_mask = valid_mask & (~np.isclose(depth, invalid_return_value, atol=atol))

    return pts[valid_mask].astype(np.float32), depth[valid_mask].astype(np.float32)


def sample_or_pad_pointcloud(points: np.ndarray, target_points: int) -> np.ndarray:
    """Force point cloud to fixed size [target_points, 3]."""
    if target_points <= 0:
        raise ValueError(f"target_points must be positive, got {target_points}")

    if points is None or points.size == 0:
        return np.zeros((target_points, 3), dtype=np.float32)

    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    n = points.shape[0]
    if n == target_points:
        return points
    if n > target_points:
        indices = np.random.choice(n, size=target_points, replace=False)
        return points[indices]

    repeat = target_points // n
    rem = target_points % n
    tiled = np.repeat(points, repeat, axis=0) if repeat > 0 else np.zeros((0, 3), dtype=np.float32)
    if rem > 0:
        extra_idx = np.random.choice(n, size=rem, replace=False)
        tiled = np.concatenate([tiled, points[extra_idx]], axis=0)
    return tiled.astype(np.float32)


def filter_robot_base_region_points(
    robot,
    env_id: int,
    points: np.ndarray,
    base_body_name: str = ROBOT_PC_BASE_FILTER_BODY,
    radius: float = ROBOT_PC_BASE_FILTER_RADIUS,
    height: float = ROBOT_PC_BASE_FILTER_HEIGHT,
) -> np.ndarray:
    """Remove points in a low cylindrical region around the robot base."""
    if points is None or points.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    body_names = list(getattr(robot, "body_names", []))
    if base_body_name not in body_names:
        return np.asarray(points, dtype=np.float32).reshape(-1, 3)

    base_body_idx = body_names.index(base_body_name)
    base_pos = robot.data.body_state_w[env_id, base_body_idx, 0:3].clone().cpu().numpy().astype(np.float32)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    rel = points - base_pos[None, :]
    radial_dist = np.linalg.norm(rel[:, :2], axis=1)
    within_base_region = (radial_dist <= radius) & (rel[:, 2] <= height)
    filtered = points[~within_base_region]

    return filtered.astype(np.float32) if filtered.size > 0 else points


def sample_or_pad_pointcloud_with_depth(points_with_depth: np.ndarray, target_points: int) -> np.ndarray:
    """Force XYZD point cloud to fixed size [target_points, 4]."""
    if target_points <= 0:
        raise ValueError(f"target_points must be positive, got {target_points}")

    if points_with_depth is None or points_with_depth.size == 0:
        return np.zeros((target_points, 4), dtype=np.float32)

    points_with_depth = np.asarray(points_with_depth, dtype=np.float32).reshape(-1, 4)
    n = points_with_depth.shape[0]
    if n == target_points:
        return points_with_depth
    if n > target_points:
        indices = np.random.choice(n, size=target_points, replace=False)
        return points_with_depth[indices]

    repeat = target_points // n
    rem = target_points % n
    tiled = np.repeat(points_with_depth, repeat, axis=0) if repeat > 0 else np.zeros((0, 4), dtype=np.float32)
    if rem > 0:
        extra_idx = np.random.choice(n, size=rem, replace=False)
        tiled = np.concatenate([tiled, points_with_depth[extra_idx]], axis=0)
    return tiled.astype(np.float32)


def keep_closest_points_by_depth_feature(
    points_with_depth: np.ndarray,
    target_points: int,
    depth_threshold: float = 0.25,
    alpha: float = 1,
) -> np.ndarray:
    """Match dataset generation with weighted sampling from the near-point pool."""
    if target_points <= 0:
        raise ValueError(f"target_points must be positive, got {target_points}")
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")

    if points_with_depth is None or points_with_depth.size == 0:
        return np.zeros((0, 4), dtype=np.float32)

    pts = np.asarray(points_with_depth, dtype=np.float32).reshape(-1, 4)
    if pts.shape[0] <= target_points:
        return pts

    depth_feature = pts[:, 3]
    sorted_idx = np.argsort(depth_feature)[::-1]
    top_idx = sorted_idx[:target_points]
    kth_depth = depth_feature[top_idx[-1]]

    if kth_depth <= depth_threshold:
        return pts[top_idx].astype(np.float32)

    near_mask = depth_feature > depth_threshold
    near_pts = pts[near_mask]
    if near_pts.shape[0] <= target_points:
        return near_pts.astype(np.float32)

    near_depth = near_pts[:, 3]
    weights = np.power(np.clip(near_depth - depth_threshold, 1e-8, None), alpha)
    weights = weights / weights.sum()
    sample_idx = np.random.choice(
        near_pts.shape[0],
        size=target_points,
        replace=False,
        p=weights,
    )
    return near_pts[sample_idx].astype(np.float32)


def preprocess_depth_to_color_intensity(depth_values: np.ndarray) -> np.ndarray:
    """Map depth to [0, 1] intensity using the same preprocessing as dataset generation."""
    if depth_values is None:
        return np.zeros((0,), dtype=np.float32)

    depth = np.asarray(depth_values, dtype=np.float32).reshape(-1)
    if depth.size == 0:
        return np.zeros((0,), dtype=np.float32)

    intensity = np.maximum(0.0, 1.0 - depth)
    positive = intensity > 0.0
    intensity[positive] = intensity[positive] ** 2
    return np.clip(intensity, 0.0, 1.0).astype(np.float32)



def collect_current_radar_points_with_depth(
    radar_collectors: Dict[int, object],
    env_id: int,
    self_filter: Optional[RobotSelfFilter] = None,
    current_joints: Optional[np.ndarray] = None,
    frame_data: Optional[dict] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect current merged world-frame radar point cloud and processed depth feature."""
    collector = radar_collectors.get(env_id)
    if collector is None:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    if frame_data is None:
        frame_data = collector.collect_frame(include_depth=True, merge=False)
    per_lidar = frame_data.get("per_lidar", {})
    if not per_lidar:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    merged_points = []
    merged_depth = []
    for lidar_data in per_lidar.values():
        points_world = np.asarray(
            lidar_data.get("points_world", np.zeros((0, 3), dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1, 3)
        points_local = np.asarray(
            lidar_data.get("points_local_fixed", np.zeros((0, 3), dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1, 3)
        depth_image = lidar_data.get("depth_image", None)
        depth_flat = None if depth_image is None else np.asarray(depth_image, dtype=np.float32).reshape(-1)
        fallback_depth = np.linalg.norm(points_local, axis=1).astype(np.float32) if points_local.size > 0 else None

        points_valid, depth_valid = filter_points_and_depth_by_return_value(
            points_world,
            depth_flat,
            fallback_depth_values=fallback_depth,
            invalid_return_value=4.0,
            atol=1e-5,
        )

        if self_filter is not None and current_joints is not None and points_valid.size > 0:
            valid_mask = self_filter.filter_self_hits(current_joints, points_valid)
            points_valid = points_valid[valid_mask]
            depth_valid = depth_valid[valid_mask]

        if points_valid.size > 0:
            merged_points.append(points_valid)
            merged_depth.append(preprocess_depth_to_color_intensity(depth_valid))

    if not merged_points:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    return (
        np.concatenate(merged_points, axis=0).astype(np.float32),
        np.concatenate(merged_depth, axis=0).astype(np.float32),
    )


def draw_raycast_radar_points(
    draw,
    frame_data: Optional[dict],
    point_size: float = RAYCAST_RADAR_POINT_SIZE,
    stride: int = RAYCAST_RADAR_VISUAL_STRIDE,
) -> None:
    """Draw current raycast radar hit points in the Isaac viewport."""
    draw.clear_points()
    if not frame_data:
        return

    per_lidar = frame_data.get("per_lidar", {})
    if not per_lidar:
        return

    colors = (
        (0.10, 0.85, 1.00, 1.0),
        (1.00, 0.55, 0.10, 1.0),
        (0.35, 1.00, 0.30, 1.0),
        (1.00, 0.25, 0.65, 1.0),
    )
    stride = max(1, int(stride))
    points_to_draw = []
    colors_to_draw = []
    sizes_to_draw = []

    for lidar_index, lidar_data in enumerate(per_lidar.values()):
        points_world = np.asarray(
            lidar_data.get("points_world", np.zeros((0, 3), dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1, 3)
        if points_world.size == 0:
            continue

        depth_image = lidar_data.get("depth_image")
        depth = None if depth_image is None else np.asarray(depth_image, dtype=np.float32).reshape(-1)
        if depth is not None and depth.shape[0] == points_world.shape[0]:
            valid = np.isfinite(depth) & (~np.isclose(depth, 4.0, atol=1e-5))
        else:
            valid = np.isfinite(points_world).all(axis=1)

        visible_points = points_world[valid][::stride]
        color = colors[lidar_index % len(colors)]
        points_to_draw.extend(tuple(map(float, point)) for point in visible_points)
        colors_to_draw.extend([color] * len(visible_points))
        sizes_to_draw.extend([float(point_size)] * len(visible_points))

    if points_to_draw:
        draw.draw_points(points_to_draw, colors_to_draw, sizes_to_draw)


# ==================== Data Loading ====================

def get_target_demo_group(handle: h5py.File, scene_name: str, demo_id: int, target_hdf5_path: str):
    """Return a target demo group from either legacy or merged-by-scene target HDF5 files."""
    demo_key = f"demo_{demo_id}"
    if "data" not in handle:
        raise KeyError(f"'data' group not found in target HDF5: {target_hdf5_path}")

    data_group = handle["data"]
    if demo_key in data_group:
        return data_group[demo_key]

    if scene_name in data_group and demo_key in data_group[scene_name]:
        return data_group[scene_name][demo_key]

    available = ", ".join(list(data_group.keys())[:10])
    raise KeyError(
        f"Demo {demo_key} for scene '{scene_name}' not found in target HDF5: "
        f"{target_hdf5_path}. Available data keys include: {available}"
    )


def load_demo_data(
    scene_name: str,
    demo_id: int,
    target_hdf5_path: str,
    target_index: int,
) -> dict:
    """Load one start/goal pair from the compact target HDF5."""
    target_hdf5_path = os.path.abspath(os.path.expanduser(target_hdf5_path))
    demo_key = f"demo_{demo_id}"

    with h5py.File(target_hdf5_path, "r") as f:
        demo = get_target_demo_group(f, scene_name, demo_id, target_hdf5_path)
        start_positions = np.asarray(demo["start_positions"][:], dtype=np.float32)
        goal_positions = np.asarray(demo["goal_positions"][:], dtype=np.float32)

        if target_index < 0 or target_index >= len(start_positions):
            raise IndexError(
                f"target_index={target_index} is out of range for {demo_key}. "
                f"Available pairs: {len(start_positions)}"
            )

        initial_state = start_positions[target_index]
        goal_state = goal_positions[target_index]

        trajectory_name = None
        if "trajectories_names" in demo:
            raw_name = demo["trajectories_names"][target_index]
            trajectory_name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)

    print(f"\n=== Demo {demo_id} from target HDF5 ===")
    if trajectory_name is not None:
        print(f"  Trajectory: {trajectory_name}")
    print(f"  Initial: {initial_state[:4]}...")
    print(f"  Goal: {goal_state[:4]}...")

    return {
        "initial_joints": initial_state[:7].astype(np.float32),
        "goal_joints": goal_state[:7].astype(np.float32),
        "initial_full_joint_state": initial_state.astype(np.float32),
        "trajectory_name": trajectory_name,
    }


def apply_robot_joint_state(
    robot,
    env,
    joint_state: np.ndarray,
    env_ids: List[int],
    headless: bool,
    render_sensors: bool = False,
    settle_steps: int = 10,
) -> None:
    """Write a provided joint state into simulator for the selected environments."""
    state = np.asarray(joint_state, dtype=np.float32).reshape(-1)
    env_ids_tensor = torch.tensor(env_ids, device=robot.device, dtype=torch.long)

    target_pos = robot.data.joint_pos[env_ids_tensor].clone()
    target_vel = torch.zeros_like(robot.data.joint_vel[env_ids_tensor])

    joint_dim = min(state.shape[0], target_pos.shape[1])
    target_pos[:, :joint_dim] = torch.tensor(state[:joint_dim], device=robot.device, dtype=torch.float32)

    for _ in range(settle_steps):
        robot.write_joint_state_to_sim(target_pos, target_vel, env_ids=env_ids_tensor)
        robot.set_joint_position_target(target_pos, env_ids=env_ids_tensor)
        robot.write_data_to_sim()
        env.sim.step(render=(not headless) or render_sensors)
        robot.update(dt=env.sim.get_physics_dt())


def get_target_indices_for_demo(
    target_hdf5_path: str,
    scene_name: str,
    demo_id: int,
) -> List[int]:
    """Return every target pair index for one demo."""
    target_hdf5_path = os.path.abspath(os.path.expanduser(target_hdf5_path))
    with h5py.File(target_hdf5_path, "r") as f:
        demo = get_target_demo_group(f, scene_name, demo_id, target_hdf5_path)
        pair_count = len(demo["start_positions"])

    return list(range(pair_count))


# ==================== Execution with ACT Temporal Ensembling ====================

def temporal_ensemble_action(
    action_buffer: Dict[int, List[torch.Tensor]],
    current_step: int,
    chunk_size: int,
    temperature: float = 0.1
) -> Optional[torch.Tensor]:
    """
    ACT Temporal Ensembling: Average overlapping action predictions with exponential weights.

    According to Algorithm 2 in ACT paper:
    - B[t] stores all actions predicted for timestep t from previous chunks
    - at = sum_i (wi * At[i]) / sum_i(wi), where wi = exp(-m * i)
    - i is the position within the predicted chunk (0-indexed)
    """
    if current_step not in action_buffer or len(action_buffer[current_step]) == 0:
        return None

    actions_with_positions = action_buffer[current_step]
    actions = [a for a, pos in actions_with_positions]
    positions = [pos for a, pos in actions_with_positions]

    # Stack actions: [num_predictions, action_dim]
    stacked_actions = torch.stack(actions, dim=0)

    # Compute exponential weights: wi = exp(-m * i) where i is position in chunk
    weights = torch.exp(-temperature * torch.tensor(positions, dtype=torch.float32))
    weights = weights / weights.sum()  # Normalize

    # Weighted average
    weighted_actions = stacked_actions * weights.unsqueeze(1).to(stacked_actions.device)
    ensemble_action = weighted_actions.sum(dim=0)

    return ensemble_action


def execute_and_replan(
    task: Task,
    robot,
    demo_data: dict,
    collision_checker: CollisionChecker,
    robot_pc_cache,
    radar_collectors: Dict[int, object],
    self_filter: Optional[RobotSelfFilter],
    radar_merge_frames: int,
    curobo_planner: CuroboPlanner,
    radar_debug_draw=None,
    n_obs_steps: int = 2,
) -> bool:
    """
    Execute one step with ACT-style Temporal Ensembling.

    Key changes from old implementation:
    - Query policy EVERY timestep (not just every k steps)
    - Use action buffer to store overlapping predictions
    - Apply temporal ensembling (weighted average) to get smooth actions
    """
    if task.state != State.EXECUTING:
        return False

    env_id = task.env_id
    goal_joints = demo_data["goal_joints"]
    target_ee_pose = demo_data["target_ee_pose"]
    current_joints = robot.data.joint_pos[env_id][:7].cpu().numpy()

    # Check termination: goal reached
    position_error_m, orientation_error_deg = compute_ee_goal_errors(
        current_joints,
        target_ee_pose,
        curobo_planner,
    )
    if (
        position_error_m <= GOAL_THRESHOLD_POSITION_M
        and orientation_error_deg <= GOAL_THRESHOLD_ORIENTATION_DEG
    ):
        cprint(f"\n[OK] [Env {env_id}] GOAL REACHED at step {task.total_steps}, "
               f"pos_err={position_error_m:.4f}m, rot_err={orientation_error_deg:.2f}deg, "
               f"collisions={task.collision_steps}",
               "green", attrs=["bold"])
        task.state, task.success = State.DONE, True
        return True

    # Check termination: max steps. Collision tasks are capped earlier.
    task_max_steps = get_task_max_steps(task)
    if task.total_steps >= task_max_steps:
        cprint(f"\n[FAIL] [Env {env_id}] MAX STEPS at {task.total_steps}/{task_max_steps}, "
               f"collisions={task.collision_steps}", "red")
        task.state = State.DONE
        return False

    # Tracking error collision detection
    env_ids_list = [env_id]
    env_active = torch.ones(1, dtype=torch.bool, device=collision_checker.device)
    collision_mask = collision_checker.check_tracking_error(env_ids_list, robot, env_active)
    if collision_checker._target_joints is not None:
        actual = robot.data.joint_pos[env_id, :7]
        vel = collision_checker._last_velocity.get(env_id, None)
        actual_np = actual.detach().cpu().numpy()
        distance_to_limit_all = np.minimum(
            np.abs(actual_np - FRANKA_ARM_JOINT_LIMITS[:, 0]),
            np.abs(actual_np - FRANKA_ARM_JOINT_LIMITS[:, 1]),
        )

        at_joint_limit = False
        max_joint_idx = distance_to_limit_all.argmin().item()
        if distance_to_limit_all[max_joint_idx] < JOINT_LIMIT_DEBUG_THRESHOLD:
            if vel is not None and vel[max_joint_idx].abs().item() < JOINT_LIMIT_STILL_VELOCITY_THRESHOLD:
                at_joint_limit = True

        if collision_mask[0] and not at_joint_limit:
            task.collision_steps += 1
            task.collision_detected = True

        task_max_steps = get_task_max_steps(task)
        if task.total_steps >= task_max_steps:
            cprint(f"\n[FAIL] [Env {env_id}] MAX STEPS at {task.total_steps}/{task_max_steps}, "
                   f"collisions={task.collision_steps}", "red")
            task.state = State.DONE
            return False

    # Track max tracking error
    if collision_checker._target_joints is not None:
        actual = robot.data.joint_pos[env_id, :7]
        target = collision_checker._target_joints[0, :7]
        max_err = torch.abs(actual - target).max().item()
        task.max_tracking_error = max(task.max_tracking_error, max_err)

    # Collect observations
    current_joint_all = robot.data.joint_pos[env_id].cpu().numpy()
    collector = radar_collectors.get(env_id)
    radar_frame_data = collector.collect_frame(include_depth=True, merge=False) if collector is not None else None
    if radar_debug_draw is not None:
        draw_raycast_radar_points(radar_debug_draw, radar_frame_data)
    current_radar_points, current_radar_depth = collect_current_radar_points_with_depth(
        radar_collectors,
        env_id,
        self_filter=self_filter,
        current_joints=current_joint_all,
        frame_data=radar_frame_data,
    )
    current_radar_points_with_depth = np.concatenate(
        [current_radar_points, current_radar_depth[:, None]], axis=1
    ).astype(np.float32)

    _, _, radar_history_maxlen = get_radar_merge_layout(
        radar_merge_frames,
        far_stride=RADAR_FRAME_STRIDE,
    )
    if task.radar_history is None or task.radar_history.maxlen != radar_history_maxlen:
        task.radar_history = deque(maxlen=radar_history_maxlen)
    task.radar_history.append(current_radar_points_with_depth.copy())

    merged_obstacle_points = merge_recent_pointcloud_frames_with_depth(
        task.radar_history,
        window_size=radar_merge_frames,
        frame_stride=RADAR_FRAME_STRIDE,
    )
    merged_obstacle_points = keep_closest_points_by_depth_feature(
        merged_obstacle_points,
        target_points=TOTAL_OBSTACLE_POINTS,
    )
    obstacle_pc = sample_or_pad_pointcloud_with_depth(
        merged_obstacle_points, TOTAL_OBSTACLE_POINTS
    )
    obstacle_pc_colored = add_obstacle_color_with_depth_channel(obstacle_pc, OBSTACLE_COLOR)

    robot_pc = collect_robot_pointcloud_from_usd(
        robot,
        env_id,
        robot_pc_cache,
        num_points=TOTAL_ROBOT_POINTS,
    )
    if robot_pc.size == 0:
        robot_pc = collect_robot_pointcloud_from_fk(
            robot.data.joint_pos[env_id, :7], collision_checker.fk_sampler, num_points=TOTAL_ROBOT_POINTS
        )
    if ROBOT_PC_BASE_FILTER_ENABLED:
        robot_pc = filter_robot_base_region_points(
            robot,
            env_id,
            robot_pc,
            base_body_name=ROBOT_PC_BASE_FILTER_BODY,
            radius=ROBOT_PC_BASE_FILTER_RADIUS,
            height=ROBOT_PC_BASE_FILTER_HEIGHT,
        )
    robot_pc = sample_or_pad_pointcloud(robot_pc, TOTAL_ROBOT_POINTS)
    robot_pc_colored = add_color_to_pointcloud(robot_pc, ROBOT_COLOR)
    full_pc = np.concatenate([obstacle_pc_colored, robot_pc_colored], axis=0)

    target_points = TOTAL_OBSTACLE_POINTS + TOTAL_ROBOT_POINTS
    if len(full_pc) != target_points:
        if len(full_pc) > target_points:
            full_pc = full_pc[np.random.choice(len(full_pc), target_points, replace=False)]
        else:
            padding = np.zeros((target_points - len(full_pc), 6), dtype=np.float32)
            full_pc = np.concatenate([full_pc, padding], axis=0)

    obs = {
        'point_cloud': full_pc.copy(),
        'agent_pos': current_joints[:7].copy().astype(np.float32),
    }

    if task.obs_queue is None:
        task.obs_queue = deque(maxlen=n_obs_steps)
        for _ in range(n_obs_steps):
            task.obs_queue.append(obs.copy())
    else:
        task.obs_queue.append(obs)

    # ====== ACT: Query policy EVERY timestep (Algorithm 2, Line 4-5) ======
    obs_list = list(task.obs_queue)
    obs_dict = {
        'point_cloud': torch.from_numpy(
            np.stack([o['point_cloud'] for o in obs_list])
        ).float().to(task.planner.device).unsqueeze(0),
        'agent_pos': torch.from_numpy(
            np.stack([o['agent_pos'] for o in obs_list])
        ).float().to(task.planner.device).unsqueeze(0),
        'goal': torch.from_numpy(
            goal_joints[:7]
        ).float().to(task.planner.device).unsqueeze(0),
    }
    # Predict k-step action chunk (shape: [k, action_dim])
    with torch.no_grad():
        action_chunk = task.planner.policy.predict_action(obs_dict)['action'].squeeze(0).cpu()

    task.last_action_chunk = action_chunk
    task.replan_count += 1

    # ====== ACT: Add to action buffer (Algorithm 2, Line 5) ======
    for i in range(min(N_ACTION_STEPS, action_chunk.shape[0])):
        step_t = task.total_steps + i
        if step_t not in task.action_buffer:
            task.action_buffer[step_t] = []
        task.action_buffer[step_t].append((action_chunk[i], i))

    # Clean up old action buffer entries
    keys_to_remove = [k for k in task.action_buffer.keys() if k < task.total_steps]
    for k in keys_to_remove:
        del task.action_buffer[k]

    # ====== ACT: Temporal Ensembling (Algorithm 2, Line 7) ======
    ensemble_action = temporal_ensemble_action(
        task.action_buffer, task.total_steps, N_ACTION_STEPS, TEMPERATURE
    )

    if ensemble_action is None:
        ensemble_action = action_chunk[0]

    # Execute the ensembled action
    target = robot.data.joint_pos[env_id].clone()
    target[:7] = ensemble_action
    robot.set_joint_position_target(target.unsqueeze(0), env_ids=[env_id])

    target_for_checker = target.unsqueeze(0)
    collision_checker.set_target(target_for_checker, [env_id])

    task.total_steps += 1
    return True


def apply_blue_floor_material(stage) -> None:
    """Override the default ground plane visuals while preserving its collision structure."""
    ground_paths = ["/World/GroundPlane", "/World/defaultGroundPlane", "/World/ground"]
    ground_prim = None
    for ground_path in ground_paths:
        prim = stage.GetPrimAtPath(ground_path)
        if prim and prim.IsValid():
            ground_prim = prim
            break

    if ground_prim is None:
        cprint("[blue_floor] Ground plane prim not found; skip blue floor override.", "yellow")
        return

    material_path = Sdf.Path("/World/Looks/BlueFloor")
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, material_path.AppendPath("PreviewSurface"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.25, 0.55, 1.0))
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set((0.01, 0.08, 0.25))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    bound_count = 0
    for prim in Usd.PrimRange(ground_prim):
        type_name = prim.GetTypeName()
        if type_name not in {"Mesh", "Plane"}:
            continue
        UsdShade.MaterialBindingAPI(prim).Bind(
            material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        )
        bound_count += 1

    if bound_count == 0:
        UsdShade.MaterialBindingAPI(ground_prim).Bind(
            material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        )
        bound_count = 1

    cprint(
        f"[blue_floor] Applied blue material override to {bound_count} ground prim(s) under "
        f"{ground_prim.GetPath()}",
        "cyan",
    )


# ==================== Main ====================

def main():
    # Setup environment
    os.environ["ESCAPE_START_DEMO_ID"] = str(args_cli.demo_id)
    os.environ["ESCAPE_SCENE_NAME"] = args_cli.scene_name
    os.environ["ESCAPE_NUM_CYCLES"] = str(args_cli.num_cycles)
    if args_cli.goal_ghost_transparency is not None:
        args_cli.goal_ghost_opacity = 1.0 - float(args_cli.goal_ghost_transparency)
    args_cli.goal_ghost_opacity = float(np.clip(args_cli.goal_ghost_opacity, 0.0, 1.0))
    step_headless = args_cli.headless
    render_sensors = not step_headless

    radar_debug_draw = None
    if not args_cli.headless:
        radar_debug_draw = load_debug_draw_interface()
        cprint("[main] Raycast radar point visualization enabled.", "cyan")

    if str(args_cli.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {args_cli.device}, but torch.cuda.is_available() is False.")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.sim.use_fabric = True
    env_cfg.sim.device = args_cli.device
    print(f"[main] Radar mode: use_fabric=True, device={args_cli.device}")

    actual_device = args_cli.device

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()
    robot = env.scene["robot"]
    stage = omni.usd.get_context().get_stage()
    apply_blue_floor_material(stage)

    # Stabilize
    print("\n=== Stabilizing ===")
    for _ in range(50):
        robot.write_data_to_sim()
        env.sim.step(render=render_sensors)
        robot.update(dt=env.sim.get_physics_dt())

    # Collision checker tracks policy command execution errors.
    cprint("\n=== Initializing Collision Checker ===", "cyan", attrs=["bold"])
    collision_checker = CollisionChecker(
        args_cli.collision_margin,
        actual_device,
        tracking_error_threshold=args_cli.tracking_error_threshold,
        velocity_threshold=args_cli.velocity_threshold,
        sustained_steps=args_cli.sustained_steps,
    )

    cprint("\n=== Building Robot USD PointCloud Cache ===", "cyan", attrs=["bold"])
    robot_pc_cache = build_robot_pointcloud_cache_from_usd(
        stage=stage,
        robot=robot,
        env_id=0,
        num_points=TOTAL_ROBOT_POINTS,
        exclude_body_names=ROBOT_PC_EXCLUDED_BODIES,
        body_sampling_weights=ROBOT_PC_BODY_WEIGHTS,
    )
    if robot_pc_cache.total_points > 0:
        print(
            "  "
            f"[OK] Robot USD cache ready: {robot_pc_cache.total_points} pts, "
            f"{robot_pc_cache.total_links} links, {robot_pc_cache.total_geometries} geometries"
        )
        print(
            "  "
            f"excluded_bodies={ROBOT_PC_EXCLUDED_BODIES}, "
            f"weighted_bodies={ROBOT_PC_BODY_WEIGHTS}"
        )
        if ROBOT_PC_BASE_FILTER_ENABLED:
            print(
                "  "
                f"base_region_filter=({ROBOT_PC_BASE_FILTER_BODY}, "
                f"radius={ROBOT_PC_BASE_FILTER_RADIUS}, height={ROBOT_PC_BASE_FILTER_HEIGHT})"
            )
    else:
        cprint("  Robot USD cache is empty; using FK robot point cloud.", "yellow")

    cprint("\n=== Initializing Robot Self Filter ===", "cyan", attrs=["bold"])
    sphere_yaml_path = None
    if SELF_FILTER_USE_YAML:
        if os.path.isfile(SELF_FILTER_SPHERES_YAML):
            sphere_yaml_path = SELF_FILTER_SPHERES_YAML
            print(f"  Using self-filter YAML spheres: {sphere_yaml_path}")
        else:
            cprint(
                f"  SELF_FILTER_SPHERES_YAML not found: {SELF_FILTER_SPHERES_YAML}. "
                "Using no self filtering.",
                "yellow",
            )

    self_filter = RobotSelfFilter(
        device=actual_device,
        threshold=SELF_FILTER_THRESHOLD,
        sphere_yaml_path=sphere_yaml_path,
    )
    if self_filter.initialize():
        print(f"  [OK] Self filter enabled (threshold={SELF_FILTER_THRESHOLD})")
    else:
        cprint("  Self filter init failed; using no self filtering.", "yellow")
        self_filter = None

    cprint("\n=== Initializing Radar PointCloud Collectors ===", "cyan", attrs=["bold"])
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(10):
        robot.write_data_to_sim()
        env.sim.step(render=render_sensors)
        robot.update(dt=env.sim.get_physics_dt())
    lidar_paths_to_rebuild = discover_lidars(env_prefix="/World/envs/env_0")
    if len(lidar_paths_to_rebuild) == 0:
        raise RuntimeError("No source LiDAR prims found under /World/envs/env_0.")
    disabled_native_lidars = disable_native_lidar_visuals(stage, env_prefix="/World/envs/env_0")
    print(f"  Disabled native LiDAR viewport visuals on {disabled_native_lidars} prim(s)")
    print("  Radar backend: raycast")

    radar_collectors: Dict[int, object] = {}
    for env_id in range(args_cli.num_envs):
        collector = HeadlessRaycastRadarPointCloudCollector(
            robot=robot,
            env_prefix=f"/World/envs/env_{env_id}",
            env_id=env_id,
            lidar_name_pattern=r"^Lidar\d+$",
            flip_local_y=False,
        )
        collector.initialize(lidar_names=None)
        print(
            f"  env_{env_id}: raycast_headless_lidars={len(collector.get_lidar_names())}"
        )
        radar_collectors[env_id] = collector
        print(f"  env_{env_id}: lidars={collector.get_lidar_names()}")

    # Initialize CuRobo planner (for goal adjustment)
    cprint("\n=== Initializing CuRobo ===", "cyan")
    curobo_cfg = CuroboPlannerCfg.franka_config()
    curobo_cfg.visualize_plan = False
    curobo_cfg.debug_planner = False
    curobo_cfg.world_ignore_substrings = ["/World/defaultGroundPlane", "/curobo", "/obstacle_"]
    curobo_planner = CuroboPlanner(env, robot, curobo_cfg, env_id=0)

    # Initialize ManiFlow planners
    cprint(f"\n=== Creating {args_cli.num_envs} ManiFlow planners ===", "cyan", attrs=["bold"])
    planners = {
        env_id: ManiFlowPlanner(
            env=env, robot=robot, checkpoint_path=args_cli.checkpoint_path,
            env_id=env_id, debug=False, device=actual_device,
            num_inference_steps=args_cli.num_inference_steps
        ) for env_id in range(args_cli.num_envs)
    }

    # ==================== Run cycles ====================
    total_tasks = 0
    total_success = 0
    total_collision_free = 0
    total_success_and_cf = 0

    for cycle in range(args_cli.num_cycles):
        current_demo_id = args_cli.demo_id + cycle
        os.environ["ESCAPE_DEMO_ID"] = str(current_demo_id)
        target_indices = get_target_indices_for_demo(
            args_cli.target_hdf5_path, args_cli.scene_name, current_demo_id
        )

        for target_idx in target_indices:
            env.reset()
            for p in planners.values():
                p.reset_plan()

            cprint(f"\n{'='*60}", "yellow")
            cprint(
                f"CYCLE {cycle + 1}/{args_cli.num_cycles} - Demo {current_demo_id} - Target {target_idx}",
                "yellow",
                attrs=["bold"],
            )
            cprint(f"{'='*60}", "yellow")

            demo_data = load_demo_data(
                args_cli.scene_name,
                current_demo_id,
                target_hdf5_path=args_cli.target_hdf5_path,
                target_index=target_idx,
            )
            apply_robot_joint_state(
                robot=robot,
                env=env,
                joint_state=demo_data["initial_full_joint_state"],
                env_ids=list(range(args_cli.num_envs)),
                headless=step_headless,
                render_sensors=render_sensors,
                settle_steps=10,
            )
            print(
                f"  Applied target HDF5 start state for demo_{current_demo_id}, "
                f"target_index={target_idx}"
            )
            goal_joints = demo_data['goal_joints']
            target_ee_position, target_ee_quaternion_xyzw = get_ee_pose_from_joints(
                goal_joints[:7],
                curobo_planner,
            )
            demo_data["target_ee_pose"] = np.concatenate(
                [target_ee_position, target_ee_quaternion_xyzw], axis=0
            ).astype(np.float32)

            for env_id in range(args_cli.num_envs):
                if not args_cli.headless:
                    show_goal_as_ghost(
                        stage,
                        env_id,
                        goal_joints,
                        opacity=args_cli.goal_ghost_opacity,
                        mdl_path=args_cli.goal_ghost_mdl_path,
                        mdl_material=args_cli.goal_ghost_mdl_material,
                        color=tuple(args_cli.goal_ghost_color),
                        emissive_strength=args_cli.goal_ghost_emissive_strength,
                    )

            for _ in range(10):
                robot.write_data_to_sim()
                env.sim.step(render=render_sensors)
                robot.update(dt=env.sim.get_physics_dt())

            collision_checker.reset(list(range(args_cli.num_envs)))

            tasks = {
                env_id: Task(
                    env_id=env_id,
                    state=State.EXECUTING,
                    planner=planners[env_id],
                )
                for env_id in range(args_cli.num_envs)
            }

            cprint("\n=== Executing with Tracking Error Collision Detection ===", "cyan", attrs=["bold"])
            step = 0
            while simulation_app.is_running():
                if all(t.state == State.DONE for t in tasks.values()) or step >= MAX_TASK_STEPS:
                    break

                for env_id, task in tasks.items():
                    if task.state == State.EXECUTING:
                        execute_and_replan(
                            task,
                            robot,
                            demo_data,
                            collision_checker,
                            robot_pc_cache,
                            radar_collectors,
                            self_filter,
                            RADAR_MERGE_FRAMES,
                            curobo_planner,
                            radar_debug_draw=radar_debug_draw,
                        )
                robot.write_data_to_sim()
                env.sim.step(render=render_sensors)
                robot.update(dt=env.sim.get_physics_dt())
                step += 1

                if step % 50 == 0:
                    for env_id, task in tasks.items():
                        if task.state == State.EXECUTING:
                            current = robot.data.joint_pos[env_id][:7].cpu().numpy()
                            pos_err_m, rot_err_deg = compute_ee_goal_errors(
                                current,
                                demo_data["target_ee_pose"],
                                curobo_planner,
                            )
                            print(
                                f"    Step {step} [Env {env_id}]: pos_err={pos_err_m:.4f}m, "
                                f"rot_err={rot_err_deg:.2f}deg, "
                                f"collisions={task.collision_steps}, replans={task.replan_count}"
                            )

            cprint(f"\n{'-'*60}", "cyan")
            if demo_data.get("trajectory_name") is not None:
                cprint(
                    f"Cycle {cycle + 1} Summary (Demo {current_demo_id}, {demo_data['trajectory_name']})",
                    "cyan",
                    attrs=["bold"],
                )
            else:
                cprint(f"Cycle {cycle + 1} Summary (Demo {current_demo_id})", "cyan", attrs=["bold"])
            cprint(f"{'-'*60}", "cyan")

            for env_id, task in tasks.items():
                final_joints = robot.data.joint_pos[env_id][:7].cpu().numpy()
                final_position_error_m, final_orientation_error_deg = compute_ee_goal_errors(
                    final_joints,
                    demo_data["target_ee_pose"],
                    curobo_planner,
                )
                is_collision_free = task.collision_steps == 0

                if task.success and is_collision_free:
                    status_color = "green"
                    status = "SUCCESS (collision-free)"
                elif task.success and not is_collision_free:
                    status_color = "yellow"
                    status = f"SUCCESS (with {task.collision_steps} collision steps)"
                else:
                    status_color = "red"
                    status = f"FAILED (collisions={task.collision_steps})"

                cprint(f"  [Env {env_id}] {status}", status_color, attrs=["bold"])
                print(
                    f"           steps={task.total_steps} | replans={task.replan_count} | "
                    f"pos_err={final_position_error_m:.4f}m | "
                    f"rot_err={final_orientation_error_deg:.2f}deg | "
                    f"max_tracking_err={task.max_tracking_error:.4f}"
                )

                total_tasks += 1
                total_success += int(task.success)
                total_collision_free += int(is_collision_free)
                total_success_and_cf += int(task.success and is_collision_free)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ==================== Final Summary ====================
    cprint(f"\n{'='*60}", "white")
    cprint("FINAL EVALUATION SUMMARY", "white", attrs=["bold"])
    cprint(f"{'='*60}", "white")

    print(f"  Scene: {args_cli.scene_name}")
    print(f"  Demos: {args_cli.demo_id} ~ {args_cli.demo_id + args_cli.num_cycles - 1}")
    print("  Target pairs: all trajectories from target HDF5")
    print(f"  Envs per demo: {args_cli.num_envs}")
    print(f"  Total tasks: {total_tasks}")
    cprint(f"  Goal reached:     {total_success}/{total_tasks} "
           f"({100*total_success/max(total_tasks,1):.1f}%)",
           "green" if total_success == total_tasks else "yellow", attrs=["bold"])
    cprint(f"  Collision-free:   {total_collision_free}/{total_tasks} "
           f"({100*total_collision_free/max(total_tasks,1):.1f}%)",
           "green" if total_collision_free == total_tasks else "red", attrs=["bold"])
    cprint(f"  Success + CF:     {total_success_and_cf}/{total_tasks} "
           f"({100*total_success_and_cf/max(total_tasks,1):.1f}%)",
           "green" if total_success_and_cf == total_tasks else "red", attrs=["bold"])

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
