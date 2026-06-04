#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utilities for collecting radar LiDAR point clouds from an initialized Isaac Sim stage.

The collector discovers LiDAR sensors under one environment, reads direct point
cloud returns, transforms points from each LiDAR frame into world coordinates,
and can return either per-LiDAR data or a merged robot-level point cloud.

This module intentionally does not launch the app, create environments, write
HDF5 files, render GUI visualizations, or replay trajectories. Import it after
Isaac Sim and the target stage are initialized.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Iterable, Any

import numpy as np
import omni.usd
from pxr import UsdGeom


# =========================================================
# 1) Basic helpers
# =========================================================

def acquire_default_lidar_interface():
    """
    Acquire the PhysX LiDAR interface behind a small compatibility wrapper.
    """
    from isaacsim.sensors.physx import _range_sensor
    return _range_sensor.acquire_lidar_sensor_interface()


def _call_lidar_interface(lidar_interface, method_names: Iterable[str], *args):
    """
    Call snake_case or camelCase LiDAR APIs across Isaac Sim versions.
    """
    for method_name in method_names:
        method = getattr(lidar_interface, method_name, None)
        if method is None:
            continue
        return method(*args)

    raise AttributeError(
        f"Lidar interface does not provide any of these methods: {list(method_names)}"
    )


def _gf_matrix_to_np_column_convention(gf_mat) -> np.ndarray:
    """
    Convert USD/Gf matrices to the column-vector convention used by this module.
    """
    mat = np.array([[gf_mat[i][j] for j in range(4)] for i in range(4)], dtype=np.float32)
    return mat.T


def quat_wxyz_to_rotmat(quat_wxyz: np.ndarray) -> np.ndarray:
    """
    Convert a [w, x, y, z] quaternion to a 3x3 rotation matrix.
    """
    q = np.asarray(quat_wxyz, dtype=np.float64)
    if q.shape[0] != 4:
        raise ValueError(f"quat_wxyz must have shape (4,), got {q.shape}")

    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)

    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z

    R = np.array(
        [
            [1.0 - (yy + zz), xy - wz,         xz + wy],
            [xy + wz,         1.0 - (xx + zz), yz - wx],
            [xz - wy,         yz + wx,         1.0 - (xx + yy)],
        ],
        dtype=np.float32,
    )
    return R


def transform_points_local_to_world(points_local: np.ndarray, T_local_to_world: np.ndarray) -> np.ndarray:
    """
    Transform [N, 3] local-frame points into world coordinates.
    """
    if points_local is None or points_local.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    rot = T_local_to_world[:3, :3]
    trans = T_local_to_world[:3, 3]
    return (points_local @ rot.T + trans).astype(np.float32)


def pointcloud_stats(points: np.ndarray) -> Dict[str, Any]:
    """
    Return simple point-cloud statistics for diagnostics.
    """
    if points is None or len(points) == 0:
        return {
            "n": 0,
            "centroid": None,
            "min_xyz": None,
            "max_xyz": None,
            "min_norm": None,
            "mean_norm": None,
            "max_norm": None,
        }

    norms = np.linalg.norm(points, axis=1)
    return {
        "n": int(len(points)),
        "centroid": np.mean(points, axis=0),
        "min_xyz": np.min(points, axis=0),
        "max_xyz": np.max(points, axis=0),
        "min_norm": float(np.min(norms)),
        "mean_norm": float(np.mean(norms)),
        "max_norm": float(np.max(norms)),
    }


# =========================================================
# 2) USD, prim, and link helpers
# =========================================================

def get_prim_world_transform_matrix(prim_path: str) -> np.ndarray:
    """
    Read the world transform matrix for a prim.
    """
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Invalid prim path: {prim_path}")

    xform_cache = UsdGeom.XformCache()
    gf_mat = xform_cache.GetLocalToWorldTransform(prim)
    return _gf_matrix_to_np_column_convention(gf_mat)


def get_local_transform_matrix(prim_path: str) -> np.ndarray:
    """
    Read the local transform matrix of a prim relative to its parent.
    """
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Invalid prim path: {prim_path}")

    xformable = UsdGeom.Xformable(prim)
    result = xformable.GetLocalTransformation()

    # Isaac Sim versions may return either a matrix or (matrix, resetsXformStack).
    gf_mat = result[0] if isinstance(result, tuple) else result
    return _gf_matrix_to_np_column_convention(gf_mat)


def discover_lidars(
    env_prefix: str = "/World/envs/env_0",
    lidar_name_pattern: str = r"^Lidar\d+$",
) -> Dict[str, str]:
    """
    Discover LiDAR prims under one environment prefix. Returns:
        {
            "Lidar1": "/World/envs/env_0/Robot/panda_hand/Link7_lidar/Lidar1",
            "Lidar2": "...",
            ...
        }
    """
    stage = omni.usd.get_context().get_stage()
    pattern = re.compile(lidar_name_pattern)

    lidar_paths: Dict[str, str] = {}

    for prim in stage.Traverse():
        name = prim.GetName()
        if not pattern.match(name):
            continue

        path_str = prim.GetPath().pathString
        if env_prefix not in path_str:
            continue

        lidar_paths[name] = path_str

    def sort_key(item):
        lidar_name, _ = item
        m = re.search(r"(\d+)$", lidar_name)
        return int(m.group(1)) if m else 999999

    return dict(sorted(lidar_paths.items(), key=sort_key))


def infer_moving_link_path_from_lidar_path(
    lidar_path: str,
    robot_root_keyword: str = "Robot",
) -> Tuple[str, str]:
    """
    Infer the moving robot link that owns a LiDAR prim. Example:
        /World/envs/env_0/Robot/panda_hand/Link7_lidar/Lidar2

    resolves to:
        moving_link_path = /World/envs/env_0/Robot/panda_hand
        moving_link_name = panda_hand

    The first path component after the robot root is treated as the moving link.
    """
    parts = lidar_path.strip("/").split("/")
    if robot_root_keyword not in parts:
        raise RuntimeError(
            f"Cannot infer moving link from lidar path: {lidar_path}, "
            f"keyword '{robot_root_keyword}' not found."
        )

    robot_idx = parts.index(robot_root_keyword)
    if robot_idx + 1 >= len(parts):
        raise RuntimeError(f"Malformed lidar path: {lidar_path}")

    moving_link_parts = parts[: robot_idx + 2]
    moving_link_path = "/" + "/".join(moving_link_parts)
    moving_link_name = moving_link_parts[-1]
    return moving_link_path, moving_link_name


def compute_static_transform_link_to_lidar(
    lidar_path: str,
    robot_root_keyword: str = "Robot",
) -> Tuple[str, str, np.ndarray]:
    """
    Compute the fixed extrinsic transform T_link_to_lidar. Returns:
        moving_link_path
        moving_link_name
        T_link_to_lidar
    """
    moving_link_path, moving_link_name = infer_moving_link_path_from_lidar_path(
        lidar_path=lidar_path,
        robot_root_keyword=robot_root_keyword,
    )

    lidar_parts = lidar_path.strip("/").split("/")
    link_parts = moving_link_path.strip("/").split("/")
    relative_parts = lidar_parts[len(link_parts):]

    T_link_to_lidar = np.eye(4, dtype=np.float32)
    current_path = moving_link_path

    for child_name in relative_parts:
        current_path = current_path + "/" + child_name
        T_link_to_lidar = T_link_to_lidar @ get_local_transform_matrix(current_path)

    return moving_link_path, moving_link_name, T_link_to_lidar


def get_link_world_transform_matrix(
    robot,
    moving_link_name: str,
    env_id: int = 0,
    moving_link_path: Optional[str] = None,
) -> np.ndarray:
    """
    Resolve a moving link world transform, preferring IsaacLab body_state_w and falling back to USD prim transforms.
    """
    body_id = None

    # -----------------------------------------------------
    # Try articulation.find_bodies(...) first.
    # -----------------------------------------------------
    find_bodies_fn = getattr(robot, "find_bodies", None)
    if callable(find_bodies_fn):
        try:
            body_ids = find_bodies_fn(moving_link_name)[0]
            if len(body_ids) > 0:
                body_id = int(body_ids[0])
        except Exception:
            body_id = None

    # -----------------------------------------------------
    # Fall back to body_names when needed.
    # -----------------------------------------------------
    if body_id is None:
        body_names = getattr(robot, "body_names", None)
        if body_names is not None:
            body_names = list(body_names)
            if moving_link_name in body_names:
                body_id = body_names.index(moving_link_name)

    # -----------------------------------------------------
    # Build the world transform from body_state_w when a body id is available.
    # -----------------------------------------------------
    if body_id is not None:
        body_state = robot.data.body_state_w[env_id, body_id, 0:7].clone().cpu().numpy()
        pos = body_state[:3].astype(np.float32)
        quat_wxyz = body_state[3:7].astype(np.float32)

        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = quat_wxyz_to_rotmat(quat_wxyz)
        T[:3, 3] = pos
        return T

    # -----------------------------------------------------
    # Fall back to the link prim world transform.
    # -----------------------------------------------------
    if moving_link_path is None:
        raise RuntimeError(
            f"Cannot resolve moving link world transform for '{moving_link_name}', "
            f"and no moving_link_path is provided for fallback."
        )

    return get_prim_world_transform_matrix(moving_link_path)


# =========================================================
# 3) LiDAR static data
# =========================================================

@dataclass
class LidarStaticInfo:
    """
    Static LiDAR metadata cached after initialization.
    """
    lidar_name: str
    lidar_path: str

    moving_link_name: str
    moving_link_path: str

    T_link_to_lidar: np.ndarray

    num_rows: int
    num_cols: int

    horizontal_fov_deg: Optional[float]
    vertical_fov_deg: Optional[float]
    horizontal_resolution_deg: Optional[float]
    vertical_resolution_deg: Optional[float]
    yaw_offset_deg: Optional[float]
    min_range: Optional[float]
    max_range: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a dict that is easier to print or serialize.
        """
        data = asdict(self)

        # Preserve numpy arrays explicitly.
        data["T_link_to_lidar"] = np.array(self.T_link_to_lidar, dtype=np.float32)
        return data


# =========================================================
# 4) Collector
# =========================================================

class RadarPointCloudCollector:
    """
    Robot-level LiDAR point-cloud collector with cached static sensor metadata.
    """

    def __init__(
        self,
        robot,
        lidar_interface=None,
        env_prefix: str = "/World/envs/env_0",
        env_id: int = 0,
        lidar_name_pattern: str = r"^Lidar\d+$",
        robot_root_keyword: str = "Robot",
        flip_local_y: bool = False,
    ):
        """
        Initialize the collector for one environment. Set flip_local_y for sensors whose local Y axis needs mirroring.
        """
        self.robot = robot
        self.lidar_interface = lidar_interface if lidar_interface is not None else acquire_default_lidar_interface()

        self.env_prefix = env_prefix
        self.env_id = env_id
        self.lidar_name_pattern = lidar_name_pattern
        self.robot_root_keyword = robot_root_keyword
        self.flip_local_y = flip_local_y

        # Filled by initialize().
        self.lidar_infos: Dict[str, LidarStaticInfo] = {}
        self.lidar_order: List[str] = []

    # -----------------------------------------------------
    # 4.1 Low-level readers
    # -----------------------------------------------------

    def get_lidar_depth_image(self, lidar_path: str) -> Optional[np.ndarray]:
        """
        Read the current depth image for one LiDAR, returning None when unavailable.
        """
        try:
            depth_buffer = _call_lidar_interface(
                self.lidar_interface,
                ["get_linear_depth_data", "getLinearDepthData"],
                lidar_path,
            )
            if depth_buffer is None:
                return None

            depth_buffer = np.asarray(depth_buffer, dtype=np.float32)

            num_cols = int(_call_lidar_interface(
                self.lidar_interface,
                ["get_num_cols", "getNumCols"],
                lidar_path,
            ))
            num_rows = int(_call_lidar_interface(
                self.lidar_interface,
                ["get_num_rows", "getNumRows"],
                lidar_path,
            ))

            if depth_buffer.size != num_rows * num_cols:
                raise RuntimeError(
                    f"LiDAR depth size mismatch for {lidar_path}: "
                    f"{depth_buffer.size} vs {num_rows}x{num_cols}"
                )

            return depth_buffer.reshape(num_rows, num_cols)

        except Exception:
            return None

    def get_lidar_pointcloud_direct_raw(self, lidar_path: str) -> np.ndarray:
        """
        Read direct point-cloud returns from the LiDAR interface without obstacle-hit filtering.
        """
        pc_raw = _call_lidar_interface(
            self.lidar_interface,
            ["get_point_cloud_data", "getPointCloud"],
            lidar_path,
        )

        if pc_raw is None:
            return np.zeros((0, 3), dtype=np.float32)

        pc = np.asarray(pc_raw, dtype=np.float32)
        if pc.size == 0:
            return np.zeros((0, 3), dtype=np.float32)

        # Normalize to [N, 3].
        if pc.ndim == 1:
            if pc.size % 3 != 0:
                raise RuntimeError(f"Unexpected point cloud shape for {lidar_path}: {pc.shape}")
            pc = pc.reshape(-1, 3)

        elif pc.ndim == 2:
            if pc.shape[1] == 3:
                pass
            elif pc.shape[0] == 3:
                pc = pc.T
            else:
                pc = pc.reshape(-1, 3)

        else:
            pc = pc.reshape(-1, 3)

        # Keep only finite points.
        valid = np.isfinite(pc).all(axis=1)
        return pc[valid].astype(np.float32)

    def get_lidar_static_sensor_attrs(self, lidar_path: str) -> Dict[str, Optional[float]]:
        """
        Read static LiDAR attributes used for diagnostics and optional depth reconstruction.
        """
        def safe_call(method_names):
            try:
                return _call_lidar_interface(self.lidar_interface, method_names, lidar_path)
            except Exception:
                return None

        def to_float_or_none(v):
            if v is None:
                return None
            try:
                return float(v)
            except Exception:
                return None

        def to_int_or_zero(v):
            if v is None:
                return 0
            try:
                return int(v)
            except Exception:
                return 0

        return {
            "num_rows": to_int_or_zero(safe_call(["get_num_rows", "getNumRows"])),
            "num_cols": to_int_or_zero(safe_call(["get_num_cols", "getNumCols"])),
            "horizontal_fov_deg": to_float_or_none(safe_call(["get_horizontal_fov", "getHorizontalFov"])),
            "vertical_fov_deg": to_float_or_none(safe_call(["get_vertical_fov", "getVerticalFov"])),
            "horizontal_resolution_deg": to_float_or_none(safe_call(["get_horizontal_resolution", "getHorizontalResolution"])),
            "vertical_resolution_deg": to_float_or_none(safe_call(["get_vertical_resolution", "getVerticalResolution"])),
            "yaw_offset_deg": to_float_or_none(safe_call(["get_yaw_offset", "getYawOffset"])),
            "min_range": to_float_or_none(safe_call(["get_min_range", "getMinRange"])),
            "max_range": to_float_or_none(safe_call(["get_max_range", "getMaxRange"])),
        }

    # -----------------------------------------------------
    # 4.2 Initialization
    # -----------------------------------------------------

    def initialize(self, lidar_names: Optional[Iterable[str]] = None):
        """
        Discover LiDARs, read their static attributes, and precompute fixed extrinsics.
        """
        all_lidar_paths = discover_lidars(
            env_prefix=self.env_prefix,
            lidar_name_pattern=self.lidar_name_pattern,
        )

        if len(all_lidar_paths) == 0:
            raise RuntimeError(
                f"No lidars found under env_prefix='{self.env_prefix}' "
                f"with pattern='{self.lidar_name_pattern}'."
            )

        if lidar_names is not None:
            lidar_names = list(lidar_names)
            filtered_paths = {}
            for name in lidar_names:
                if name not in all_lidar_paths:
                    raise KeyError(
                        f"Requested lidar '{name}' not found. "
                        f"Available lidars: {list(all_lidar_paths.keys())}"
                    )
                filtered_paths[name] = all_lidar_paths[name]
            lidar_paths = filtered_paths
        else:
            lidar_paths = all_lidar_paths

        lidar_infos: Dict[str, LidarStaticInfo] = {}

        for lidar_name, lidar_path in lidar_paths.items():
            moving_link_path, moving_link_name, T_link_to_lidar = compute_static_transform_link_to_lidar(
                lidar_path=lidar_path,
                robot_root_keyword=self.robot_root_keyword,
            )

            sensor_attrs = self.get_lidar_static_sensor_attrs(lidar_path)

            lidar_infos[lidar_name] = LidarStaticInfo(
                lidar_name=lidar_name,
                lidar_path=lidar_path,
                moving_link_name=moving_link_name,
                moving_link_path=moving_link_path,
                T_link_to_lidar=T_link_to_lidar.astype(np.float32),

                num_rows=int(sensor_attrs["num_rows"]),
                num_cols=int(sensor_attrs["num_cols"]),
                horizontal_fov_deg=sensor_attrs["horizontal_fov_deg"],
                vertical_fov_deg=sensor_attrs["vertical_fov_deg"],
                horizontal_resolution_deg=sensor_attrs["horizontal_resolution_deg"],
                vertical_resolution_deg=sensor_attrs["vertical_resolution_deg"],
                yaw_offset_deg=sensor_attrs["yaw_offset_deg"],
                min_range=sensor_attrs["min_range"],
                max_range=sensor_attrs["max_range"],
            )

        self.lidar_infos = lidar_infos
        self.lidar_order = list(lidar_infos.keys())

    # -----------------------------------------------------
    # 4.3 Public helpers
    # -----------------------------------------------------

    def is_initialized(self) -> bool:
        """
        Return whether the collector has been initialized.
        """
        return len(self.lidar_infos) > 0

    def get_lidar_names(self) -> List[str]:
        """
        Return initialized LiDAR names.
        """
        return list(self.lidar_order)

    def get_static_info_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        Return static LiDAR metadata for printing or serialization.
        """
        if not self.is_initialized():
            raise RuntimeError("RadarPointCloudCollector is not initialized. Call initialize() first.")

        return {
            name: info.to_dict()
            for name, info in self.lidar_infos.items()
        }

    # -----------------------------------------------------
    # 4.4 Frame collection
    # -----------------------------------------------------

    def collect_frame(
        self,
        include_depth: bool = False,
        merge: bool = True,
        selected_lidars: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        """
        Collect the current frame from selected LiDARs and optionally merge world-frame points.
        """
        if not self.is_initialized():
            raise RuntimeError("RadarPointCloudCollector is not initialized. Call initialize() first.")

        if selected_lidars is None:
            lidar_names = self.lidar_order
        else:
            lidar_names = list(selected_lidars)
            for name in lidar_names:
                if name not in self.lidar_infos:
                    raise KeyError(
                        f"Requested lidar '{name}' is not initialized. "
                        f"Available initialized lidars: {self.lidar_order}"
                    )

        per_lidar: Dict[str, Dict[str, Any]] = {}

        merged_points_world_list: List[np.ndarray] = []
        merged_lidar_ids_list: List[np.ndarray] = []
        merged_lidar_names: List[str] = []

        for lidar_idx, lidar_name in enumerate(lidar_names):
            info = self.lidar_infos[lidar_name]

            # -------------------------------------------------
            # Read local raw point cloud.
            # -------------------------------------------------
            points_local_raw = self.get_lidar_pointcloud_direct_raw(info.lidar_path)

            # -------------------------------------------------
            # Apply local-frame correction when requested.
            # -------------------------------------------------
            points_local_fixed = points_local_raw.copy()
            if self.flip_local_y and points_local_fixed.size > 0:
                points_local_fixed[:, 1] *= -1.0

            # -------------------------------------------------
            # Read the link world transform.
            # -------------------------------------------------
            T_link_to_world = get_link_world_transform_matrix(
                robot=self.robot,
                moving_link_name=info.moving_link_name,
                env_id=self.env_id,
                moving_link_path=info.moving_link_path,
            )

            # -------------------------------------------------
            # Build the LiDAR world transform.
            # -------------------------------------------------
            T_lidar_to_world = (T_link_to_world @ info.T_link_to_lidar).astype(np.float32)

            # -------------------------------------------------
            # 5) local -> world
            # -------------------------------------------------
            points_world = transform_points_local_to_world(points_local_fixed, T_lidar_to_world)

            # -------------------------------------------------
            # Optionally read depth.
            # -------------------------------------------------
            depth_image = self.get_lidar_depth_image(info.lidar_path) if include_depth else None

            # -------------------------------------------------
            # Build per-LiDAR output.
            # -------------------------------------------------
            per_lidar[lidar_name] = {
                "points_local_raw": points_local_raw,
                "points_local_fixed": points_local_fixed,
                "points_world": points_world,
                "depth_image": depth_image,

                "lidar_path": info.lidar_path,
                "moving_link_name": info.moving_link_name,
                "moving_link_path": info.moving_link_path,

                "T_link_to_lidar": info.T_link_to_lidar.copy(),
                "T_link_to_world": T_link_to_world,
                "T_lidar_to_world": T_lidar_to_world,

                "num_rows": info.num_rows,
                "num_cols": info.num_cols,
                "horizontal_fov_deg": info.horizontal_fov_deg,
                "vertical_fov_deg": info.vertical_fov_deg,
                "horizontal_resolution_deg": info.horizontal_resolution_deg,
                "vertical_resolution_deg": info.vertical_resolution_deg,
                "yaw_offset_deg": info.yaw_offset_deg,
                "min_range": info.min_range,
                "max_range": info.max_range,
            }

            # -------------------------------------------------
            # Build merged point cloud.
            # -------------------------------------------------
            if merge and points_world.size > 0:
                merged_points_world_list.append(points_world)

                # Attach the source LiDAR id to each point.
                merged_lidar_ids_list.append(
                    np.full((points_world.shape[0],), lidar_idx, dtype=np.int32)
                )

            if merge:
                merged_lidar_names.append(lidar_name)

        # -----------------------------------------------------
        # Concatenate merged data.
        # -----------------------------------------------------
        if merge:
            if len(merged_points_world_list) > 0:
                merged_points_world = np.concatenate(merged_points_world_list, axis=0).astype(np.float32)
                merged_lidar_ids = np.concatenate(merged_lidar_ids_list, axis=0).astype(np.int32)
            else:
                merged_points_world = np.zeros((0, 3), dtype=np.float32)
                merged_lidar_ids = np.zeros((0,), dtype=np.int32)
        else:
            merged_points_world = None
            merged_lidar_ids = None
            merged_lidar_names = []

        return {
            "per_lidar": per_lidar,
            "merged_points_world": merged_points_world,
            "merged_lidar_ids": merged_lidar_ids,
            "merged_lidar_names": merged_lidar_names,
        }

    # -----------------------------------------------------
    # 4.5 Convenience helpers
    # -----------------------------------------------------

    def collect_merged_world_points(
        self,
        selected_lidars: Optional[Iterable[str]] = None,
    ) -> np.ndarray:
        """
        Return only the merged robot-level world-frame point cloud.
        """
        frame = self.collect_frame(
            include_depth=False,
            merge=True,
            selected_lidars=selected_lidars,
        )
        return frame["merged_points_world"]

    def collect_per_lidar_world_points(
        self,
        selected_lidars: Optional[Iterable[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Return per-LiDAR world-frame point clouds.
        """
        frame = self.collect_frame(
            include_depth=False,
            merge=False,
            selected_lidars=selected_lidars,
        )
        return {
            name: data["points_world"]
            for name, data in frame["per_lidar"].items()
        }

    def summarize_current_frame(
        self,
        selected_lidars: Optional[Iterable[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Return simple per-LiDAR point-cloud statistics for the current frame.
        """
        frame = self.collect_frame(
            include_depth=False,
            merge=False,
            selected_lidars=selected_lidars,
        )

        summary = {}
        for lidar_name, data in frame["per_lidar"].items():
            summary[lidar_name] = {
                "local_raw": pointcloud_stats(data["points_local_raw"]),
                "local_fixed": pointcloud_stats(data["points_local_fixed"]),
                "world": pointcloud_stats(data["points_world"]),
                "lidar_path": data["lidar_path"],
                "moving_link_name": data["moving_link_name"],
                "moving_link_path": data["moving_link_path"],
            }
        return summary
