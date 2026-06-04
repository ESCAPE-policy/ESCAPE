#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Headless-only radar helpers.

This module intentionally keeps the experimental headless LiDAR registration
logic out of radar_pointcloud_utils.py, which is the GUI/original interface.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import omni.usd
import omni.physx
from pxr import Gf, UsdGeom

from ESCAPE.utils.radar_pointcloud_utils import (
    LidarStaticInfo,
    RadarPointCloudCollector,
    _call_lidar_interface,
    compute_static_transform_link_to_lidar,
    discover_lidars,
    get_link_world_transform_matrix,
    get_prim_world_transform_matrix,
    pointcloud_stats,
    transform_points_local_to_world,
)


def _get_prim_attr_number(prim, candidate_names: Iterable[str]) -> Optional[float]:
    for name in candidate_names:
        attr = prim.GetAttribute(name)
        if not attr.IsValid():
            continue
        value = attr.Get()
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return None


def _get_prim_attr_range(prim, candidate_names: Iterable[str]) -> Optional[List[float]]:
    for name in candidate_names:
        attr = prim.GetAttribute(name)
        if not attr.IsValid():
            continue
        value = attr.Get()
        if value is None:
            continue
        try:
            values = list(value)
        except Exception:
            continue
        if len(values) != 2:
            continue
        try:
            return [float(values[0]), float(values[1])]
        except Exception:
            continue
    return None


def _maybe_rad_to_deg(values: Optional[List[float]]) -> Optional[List[float]]:
    if values is None:
        return None
    if max(abs(values[0]), abs(values[1])) < (2.0 * np.pi + 0.5):
        return [float(np.rad2deg(values[0])), float(np.rad2deg(values[1]))]
    return values


def _compute_num_beams(fov_deg: Optional[float], resolution_deg: Optional[float], default_count: int) -> int:
    if fov_deg is None or resolution_deg is None or abs(float(resolution_deg)) < 1e-8:
        return int(default_count)
    return max(1, int(round(abs(float(fov_deg)) / abs(float(resolution_deg)))) + 1)


def _read_lidar_static_attrs_from_usd(
    lidar_path: str,
    default_num_rows: int = 7,
    default_num_cols: int = 7,
) -> Dict[str, Optional[float]]:
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(lidar_path)
    if not prim.IsValid():
        raise RuntimeError(f"Invalid lidar prim path: {lidar_path}")

    horizontal_fov_deg = _get_prim_attr_number(prim, ["horizontalFov", "rangeSensor:horizontalFov"])
    vertical_fov_deg = _get_prim_attr_number(prim, ["verticalFov", "rangeSensor:verticalFov"])
    horizontal_resolution_deg = _get_prim_attr_number(
        prim, ["horizontalResolution", "rangeSensor:horizontalResolution"]
    )
    vertical_resolution_deg = _get_prim_attr_number(
        prim, ["verticalResolution", "rangeSensor:verticalResolution"]
    )
    yaw_offset_deg = _get_prim_attr_number(prim, ["yawOffset", "rangeSensor:yawOffset"])
    min_range = _get_prim_attr_number(prim, ["minRange", "rangeSensor:minRange"])
    max_range = _get_prim_attr_number(prim, ["maxRange", "rangeSensor:maxRange"])
    azimuth_range_deg = _maybe_rad_to_deg(_get_prim_attr_range(prim, ["azimuthRange", "rangeSensor:azimuthRange"]))
    zenith_range_deg = _maybe_rad_to_deg(_get_prim_attr_range(prim, ["zenithRange", "rangeSensor:zenithRange"]))

    if horizontal_fov_deg is None and azimuth_range_deg is not None:
        horizontal_fov_deg = abs(float(azimuth_range_deg[1]) - float(azimuth_range_deg[0]))
    if vertical_fov_deg is None and zenith_range_deg is not None:
        vertical_fov_deg = abs(float(zenith_range_deg[1]) - float(zenith_range_deg[0]))
    if horizontal_fov_deg is None:
        horizontal_fov_deg = 90.0
    if vertical_fov_deg is None:
        vertical_fov_deg = 90.0
    if horizontal_resolution_deg is None:
        horizontal_resolution_deg = 15.0
    if vertical_resolution_deg is None:
        vertical_resolution_deg = 15.0

    num_cols = _compute_num_beams(horizontal_fov_deg, horizontal_resolution_deg, default_num_cols)
    num_rows = _compute_num_beams(vertical_fov_deg, vertical_resolution_deg, default_num_rows)

    if yaw_offset_deg is None:
        yaw_offset_deg = 0.0
    if min_range is None:
        min_range = 0.01
    if max_range is None:
        max_range = 4.0

    return {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "horizontal_fov_deg": float(horizontal_fov_deg),
        "vertical_fov_deg": float(vertical_fov_deg),
        "horizontal_resolution_deg": float(horizontal_resolution_deg),
        "vertical_resolution_deg": float(vertical_resolution_deg),
        "yaw_offset_deg": float(yaw_offset_deg),
        "min_range": float(min_range),
        "max_range": float(max_range),
        "azimuth_range_deg": azimuth_range_deg,
        "zenith_range_deg": zenith_range_deg,
    }


def _build_ray_dirs_from_attrs(attrs: Dict[str, Optional[float]]) -> np.ndarray:
    h = int(attrs["num_rows"])
    w = int(attrs["num_cols"])
    azimuth_range = attrs.get("azimuth_range_deg")
    zenith_range = attrs.get("zenith_range_deg")
    yaw_offset = float(attrs.get("yaw_offset_deg") or 0.0)

    if azimuth_range is not None:
        az = np.linspace(float(azimuth_range[0]), float(azimuth_range[1]), w, dtype=np.float32)
    else:
        hres = float(attrs["horizontal_resolution_deg"])
        az = (np.arange(w, dtype=np.float32) - 0.5 * (w - 1)) * hres + yaw_offset

    if zenith_range is not None:
        ze = np.linspace(float(zenith_range[1]), float(zenith_range[0]), h, dtype=np.float32)
    else:
        vres = float(attrs["vertical_resolution_deg"])
        ze = (0.5 * (h - 1) - np.arange(h, dtype=np.float32)) * vres

    az_grid, ze_grid = np.meshgrid(az, ze)
    az_rad = np.deg2rad(az_grid)
    ze_rad = np.deg2rad(ze_grid)

    x = np.cos(ze_rad) * np.cos(az_rad)
    y = np.cos(ze_rad) * np.sin(az_rad)
    z = np.sin(ze_rad)
    dirs = np.stack([x, y, z], axis=-1).astype(np.float32)
    dirs = dirs / np.clip(np.linalg.norm(dirs, axis=-1, keepdims=True), 1e-8, None)
    return dirs.astype(np.float32)


def _as_gf_vec3(vec: np.ndarray) -> Gf.Vec3f:
    vec = np.asarray(vec, dtype=np.float32).reshape(3)
    return Gf.Vec3f(float(vec[0]), float(vec[1]), float(vec[2]))


def _usd_matrix_to_numpy_column_convention(gf_mat) -> np.ndarray:
    mat = np.array([[gf_mat[i][j] for j in range(4)] for i in range(4)], dtype=np.float32)
    return mat.T


def _transform_points_world(points_local: np.ndarray, world_transform) -> np.ndarray:
    points_local = np.asarray(points_local, dtype=np.float32).reshape(-1, 3)
    if points_local.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    mat = _usd_matrix_to_numpy_column_convention(world_transform)
    points_h = np.concatenate(
        [points_local, np.ones((points_local.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    return (mat @ points_h.T).T[:, :3].astype(np.float32)


def _triangulate_faces(face_vertex_counts, face_vertex_indices) -> np.ndarray:
    faces = []
    cursor = 0
    for count in face_vertex_counts:
        count = int(count)
        if count == 3:
            faces.append(face_vertex_indices[cursor:cursor + 3])
        elif count > 3:
            for i in range(1, count - 1):
                faces.append(
                    [
                        face_vertex_indices[cursor],
                        face_vertex_indices[cursor + i],
                        face_vertex_indices[cursor + i + 1],
                    ]
                )
        cursor += count
    if len(faces) == 0:
        return np.zeros((0, 3), dtype=np.int64)
    return np.asarray(faces, dtype=np.int64)


def _cube_local_mesh_from_prim(prim) -> Tuple[np.ndarray, np.ndarray]:
    size_attr = prim.GetAttribute("size")
    size = float(size_attr.Get() if size_attr and size_attr.HasAuthoredValue() else 2.0)
    half = 0.5 * size
    vertices = np.asarray(
        [
            [-half, -half, -half],
            [half, -half, -half],
            [half, half, -half],
            [-half, half, -half],
            [-half, -half, half],
            [half, -half, half],
            [half, half, half],
            [-half, half, half],
        ],
        dtype=np.float32,
    )
    faces = np.asarray(
        [
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1],
            [1, 5, 6], [1, 6, 2],
            [2, 6, 7], [2, 7, 3],
            [3, 7, 4], [3, 4, 0],
        ],
        dtype=np.int64,
    )
    return vertices, faces


def _prim_to_world_triangles(prim) -> np.ndarray:
    prim_type = prim.GetTypeName()
    if prim_type == "Mesh":
        mesh = UsdGeom.Mesh(prim)
        vertices = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if vertices is None or counts is None or indices is None:
            return np.zeros((0, 3, 3), dtype=np.float32)
        vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
        faces = _triangulate_faces(np.asarray(counts), np.asarray(indices))
    elif prim_type == "Cube":
        vertices, faces = _cube_local_mesh_from_prim(prim)
    else:
        return np.zeros((0, 3, 3), dtype=np.float32)

    if vertices.size == 0 or faces.size == 0:
        return np.zeros((0, 3, 3), dtype=np.float32)

    vertices_world = _transform_points_world(
        vertices,
        UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0),
    )
    return vertices_world[faces].astype(np.float32)


def _ray_triangle_nearest_distance(
    origin: np.ndarray,
    direction: np.ndarray,
    triangles: np.ndarray,
    max_range: float,
) -> float:
    triangles = np.asarray(triangles, dtype=np.float32).reshape(-1, 3, 3)
    if triangles.size == 0:
        return float(max_range)

    origin = np.asarray(origin, dtype=np.float32).reshape(3)
    direction = np.asarray(direction, dtype=np.float32).reshape(3)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-8:
        return float(max_range)
    direction = direction / norm

    v0 = triangles[:, 0, :]
    v1 = triangles[:, 1, :]
    v2 = triangles[:, 2, :]
    edge1 = v1 - v0
    edge2 = v2 - v0

    pvec = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
    det = np.einsum("ij,ij->i", edge1, pvec)
    det_mask = np.abs(det) > 1e-8
    if not np.any(det_mask):
        return float(max_range)

    inv_det = np.zeros_like(det, dtype=np.float32)
    inv_det[det_mask] = 1.0 / det[det_mask]

    tvec = origin[None, :] - v0
    u = np.einsum("ij,ij->i", tvec, pvec) * inv_det
    u_mask = (u >= 0.0) & (u <= 1.0) & det_mask
    if not np.any(u_mask):
        return float(max_range)

    qvec = np.cross(tvec, edge1)
    v = np.einsum("j,ij->i", direction, qvec) * inv_det
    uv_mask = u_mask & (v >= 0.0) & ((u + v) <= 1.0)
    if not np.any(uv_mask):
        return float(max_range)

    distance = np.einsum("ij,ij->i", edge2, qvec) * inv_det
    hit_mask = uv_mask & (distance > 1e-5) & (distance < float(max_range))
    if not np.any(hit_mask):
        return float(max_range)
    return float(np.min(distance[hit_mask]))


def _looks_like_obstacle_prim(path: str) -> bool:
    lower = path.lower()
    if "/robot/cube" in lower:
        return True
    if "/robot/" in lower:
        return False
    if "/runtime" in lower or "/visuals/" in lower or "/curobo" in lower:
        return False
    if "/obstacles" in lower or "obstacle" in lower:
        return True
    return False


def _collect_usd_obstacle_triangles(env_prefix: str) -> Tuple[np.ndarray, List[str]]:
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(env_prefix)
    if not root.IsValid():
        return np.zeros((0, 3, 3), dtype=np.float32), []

    triangles_list: List[np.ndarray] = []
    source_paths: List[str] = []
    stack = list(root.GetChildren())
    while stack:
        prim = stack.pop()
        stack.extend(list(prim.GetChildren()))
        if prim.GetTypeName() not in {"Mesh", "Cube"}:
            continue

        path = prim.GetPath().pathString
        if not _looks_like_obstacle_prim(path):
            continue

        triangles = _prim_to_world_triangles(prim)
        if triangles.size == 0:
            continue
        triangles_list.append(triangles)
        source_paths.append(path)

    if len(triangles_list) == 0:
        return np.zeros((0, 3, 3), dtype=np.float32), []
    return np.concatenate(triangles_list, axis=0).astype(np.float32), source_paths


class HeadlessRobotSelfFilter:
    """Headless-safe copy of the radar self-hit filter.

    The GUI helper module imports the native PhysX RangeSensor interface at
    module import time. Headless raycast inference should not depend on that
    interface, so this class keeps only the SDF/self-filtering behavior here.
    """

    def __init__(self, device: str = "cuda:0", threshold: float = 0.03, sphere_yaml_path: Optional[str] = None):
        self.device = device
        self.threshold = threshold
        self.sphere_yaml_path = sphere_yaml_path
        self.fk_sampler = None
        self.robot_urdf = None
        self.yaml_collision_spheres = None
        self._use_yaml_spheres = sphere_yaml_path is not None
        self._initialized = False

    def initialize(self) -> bool:
        if self._initialized:
            return True

        try:
            if self._use_yaml_spheres:
                from ESCAPE.robofin.robot_constants import FrankaConstants
                from ESCAPE.robofin.torch_urdf import TorchURDF

                self.yaml_collision_spheres = self._load_collision_spheres_from_yaml(self.sphere_yaml_path)
                self.robot_urdf = TorchURDF.load(
                    FrankaConstants.urdf,
                    lazy_load_meshes=True,
                    device=self.device,
                )
                sphere_count = sum(v["centers"].shape[0] for v in self.yaml_collision_spheres.values())
                print(f"[OK] HeadlessRobotSelfFilter initialized with YAML spheres: {sphere_count} spheres")
            else:
                from ESCAPE.robofin.samplers import TorchFrankaSampler

                self.fk_sampler = TorchFrankaSampler(
                    num_robot_points=1024,
                    num_eef_points=256,
                    device=self.device,
                    with_base_link=False,
                    use_cache=True,
                )
                print("[OK] HeadlessRobotSelfFilter initialized with 1024+256 sphere points")

            self._initialized = True
            return True
        except Exception as exc:
            print(f"[WARN] Failed to initialize HeadlessRobotSelfFilter: {exc}")
            self._initialized = False
            return False

    def _load_collision_spheres_from_yaml(self, yaml_path: Optional[str]) -> Dict[str, Dict[str, Any]]:
        if yaml_path is None:
            raise ValueError("sphere_yaml_path is None")

        import torch
        import yaml

        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        collision_spheres = data.get("collision_spheres", None)
        if collision_spheres is None:
            raise ValueError(f"'collision_spheres' not found in {yaml_path}")

        parsed = {}
        for link_name, sphere_list in collision_spheres.items():
            if not sphere_list:
                continue

            centers = []
            radii = []
            for sphere in sphere_list:
                center = sphere.get("center")
                radius = sphere.get("radius")
                if center is None or radius is None:
                    continue
                centers.append(center)
                radii.append(float(radius))

            if len(centers) == 0:
                continue

            parsed[link_name] = {
                "centers": torch.tensor(centers, dtype=torch.float32, device=self.device),
                "radii": torch.tensor(radii, dtype=torch.float32, device=self.device),
            }

        if len(parsed) == 0:
            raise ValueError(f"No valid spheres parsed from {yaml_path}")
        return parsed

    def _compute_robot_sdf_from_yaml_spheres(self, joint_positions, query_points):
        import torch

        if joint_positions.dim() == 1:
            joint_positions = joint_positions.unsqueeze(0)
        if query_points.dim() == 2:
            query_points = query_points.unsqueeze(0)

        batch_size = joint_positions.shape[0]
        num_points = query_points.shape[1]

        expected_dof = len(self.robot_urdf.actuated_joints)
        current_dof = joint_positions.shape[1]
        if current_dof < expected_dof:
            pad = 0.04 * torch.ones(
                (batch_size, expected_dof - current_dof),
                dtype=joint_positions.dtype,
                device=joint_positions.device,
            )
            cfg = torch.cat([joint_positions, pad], dim=1)
        elif current_dof > expected_dof:
            cfg = joint_positions[:, :expected_dof]
        else:
            cfg = joint_positions

        fk = self.robot_urdf.visual_geometry_fk_batch(cfg, use_names=True)

        all_centers_world = []
        all_radii = []
        for link_name, sphere_info in self.yaml_collision_spheres.items():
            if link_name not in fk:
                continue

            transform = fk[link_name]
            centers_local = sphere_info["centers"].unsqueeze(0).expand(batch_size, -1, -1)
            radii = sphere_info["radii"].unsqueeze(0).expand(batch_size, -1)

            rot = transform[:, :3, :3]
            trans = transform[:, :3, 3]
            centers_world = torch.matmul(centers_local, rot.transpose(1, 2)) + trans[:, None, :]

            all_centers_world.append(centers_world)
            all_radii.append(radii)

        if len(all_centers_world) == 0:
            return torch.ones((batch_size, num_points), dtype=torch.float32, device=self.device)

        centers_world = torch.cat(all_centers_world, dim=1)
        radii = torch.cat(all_radii, dim=1)
        dist_to_centers = torch.norm(query_points.unsqueeze(2) - centers_world.unsqueeze(1), dim=-1)
        sdf_to_spheres = dist_to_centers - radii.unsqueeze(1)
        min_sdf = sdf_to_spheres.min(dim=-1).values

        if batch_size == 1:
            min_sdf = min_sdf.squeeze(0)
        return min_sdf

    def compute_robot_sdf(self, joint_positions, query_points):
        import torch

        if self._use_yaml_spheres:
            if not self._initialized or self.robot_urdf is None or self.yaml_collision_spheres is None:
                if query_points.dim() == 2:
                    return torch.ones(query_points.shape[0], device=self.device) * 1.0
                return torch.ones(query_points.shape[0], query_points.shape[1], device=self.device) * 1.0
            return self._compute_robot_sdf_from_yaml_spheres(joint_positions, query_points)

        if not self._initialized or self.fk_sampler is None:
            if query_points.dim() == 2:
                return torch.ones(query_points.shape[0], device=self.device) * 1.0
            return torch.ones(query_points.shape[0], query_points.shape[1], device=self.device) * 1.0

        if joint_positions.dim() == 1:
            joint_positions = joint_positions.unsqueeze(0)
        if query_points.dim() == 2:
            query_points = query_points.unsqueeze(0)

        prismatic_joint = 0.04
        robot_pc_raw = self.fk_sampler.sample(joint_positions[:, :7], prismatic_joint)
        robot_pc = robot_pc_raw[..., :3]
        sphere_radius = 0.02

        query_expanded = query_points.unsqueeze(2)
        robot_expanded = robot_pc.unsqueeze(1)

        dist_to_centers = torch.norm(query_expanded - robot_expanded, dim=-1)
        sdf_to_spheres = dist_to_centers - sphere_radius
        min_sdf = sdf_to_spheres.min(dim=-1).values

        if joint_positions.shape[0] == 1:
            min_sdf = min_sdf.squeeze(0)
        return min_sdf

    def filter_self_hits(self, joint_positions: np.ndarray, hit_points: np.ndarray) -> np.ndarray:
        if not self._initialized:
            if not self.initialize():
                return np.ones(len(hit_points), dtype=bool)

        if len(hit_points) == 0:
            return np.ones(0, dtype=bool)

        try:
            import torch

            joints_tensor = torch.tensor(joint_positions[:7], dtype=torch.float32, device=self.device)
            points_tensor = torch.tensor(hit_points, dtype=torch.float32, device=self.device)

            sdf = self.compute_robot_sdf(joints_tensor, points_tensor)
            return (sdf.cpu().numpy() > self.threshold).astype(bool)
        except Exception as exc:
            print(f"[WARN] Headless self-filter error: {exc}, returning all points as valid")
            return np.ones(len(hit_points), dtype=bool)


def _np_column_convention_to_gf_matrix(mat: np.ndarray) -> Gf.Matrix4d:
    """Convert radar_pointcloud_utils.py's numpy convention back to Gf."""
    gf_mat = np.asarray(mat, dtype=np.float64).T
    return Gf.Matrix4d(
        gf_mat[0, 0], gf_mat[0, 1], gf_mat[0, 2], gf_mat[0, 3],
        gf_mat[1, 0], gf_mat[1, 1], gf_mat[1, 2], gf_mat[1, 3],
        gf_mat[2, 0], gf_mat[2, 1], gf_mat[2, 2], gf_mat[2, 3],
        gf_mat[3, 0], gf_mat[3, 1], gf_mat[3, 2], gf_mat[3, 3],
    )


def _ensure_xform_path(stage, path: str) -> None:
    """Create missing ancestor Xforms for a USD path."""
    current = ""
    for part in path.strip("/").split("/"):
        current = f"{current}/{part}"
        if not stage.GetPrimAtPath(current).IsValid():
            UsdGeom.Xform.Define(stage, current)


def create_runtime_lidar_prims_for_range_sensor(
    lidar_paths: Dict[str, str],
    min_range: float = 0.01,
    max_range: float = 4.0,
    horizontal_fov: float = 90.0,
    vertical_fov: float = 90.0,
    horizontal_resolution: float = 15.0,
    vertical_resolution: float = 15.0,
    rotation_rate: float = 0.0,
    high_lod: bool = False,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Create headless runtime LiDAR prims outside the referenced robot USD."""
    import omni.isaac.RangeSensorSchema as RangeSensorSchema

    stage = omni.usd.get_context().get_stage()
    runtime_paths: Dict[str, str] = {}
    source_paths: Dict[str, str] = {}

    for lidar_name, lidar_path in lidar_paths.items():
        source_world_transform = get_prim_world_transform_matrix(lidar_path)
        env_prefix = lidar_path.split("/Robot/", 1)[0]
        env_name = env_prefix.rstrip("/").split("/")[-1]
        suffix = "".join(ch for ch in lidar_name if ch.isdigit()) or str(len(runtime_paths))
        runtime_name = f"RuntimeLidar{suffix}"
        runtime_parent = f"/World/RuntimeLidars/{env_name}"
        runtime_path = f"{runtime_parent}/{runtime_name}"

        _ensure_xform_path(stage, runtime_parent)
        if stage.GetPrimAtPath(runtime_path).IsValid():
            stage.RemovePrim(runtime_path)

        try:
            lidar = RangeSensorSchema.Lidar.Define(stage, runtime_path)
            lidar_prim = lidar.GetPrim()
            range_sensor = RangeSensorSchema.RangeSensor(lidar_prim)
            range_sensor.CreateEnabledAttr(True)
            range_sensor.CreateDrawPointsAttr(False)
            range_sensor.CreateDrawLinesAttr(False)
            range_sensor.CreateMinRangeAttr(min_range)
            range_sensor.CreateMaxRangeAttr(max_range)
            lidar.CreateHorizontalFovAttr().Set(horizontal_fov)
            lidar.CreateVerticalFovAttr().Set(vertical_fov)
            lidar.CreateRotationRateAttr().Set(rotation_rate)
            lidar.CreateHorizontalResolutionAttr().Set(horizontal_resolution)
            lidar.CreateVerticalResolutionAttr().Set(vertical_resolution)
            lidar.CreateHighLodAttr().Set(high_lod)
            lidar.CreateYawOffsetAttr().Set(0.0)
            lidar.CreateEnableSemanticsAttr().Set(False)
        except Exception as exc:
            print(f"[HeadlessRadar] Failed to create runtime LiDAR for {lidar_name}: {exc}")
            continue

        if not lidar_prim.IsValid():
            print(f"[HeadlessRadar] Failed to create runtime LiDAR for {lidar_name}: {lidar_path}")
            continue

        xformable = UsdGeom.Xformable(lidar_prim)
        xformable.ClearXformOpOrder()
        xformable.AddTransformOp().Set(_np_column_convention_to_gf_matrix(source_world_transform))

        runtime_paths[runtime_name] = lidar_prim.GetPath().pathString
        source_paths[runtime_name] = lidar_path

    print(f"[HeadlessRadar] Created {len(runtime_paths)} runtime LiDAR prims")
    return runtime_paths, source_paths


class HeadlessRadarPointCloudCollector(RadarPointCloudCollector):
    """Radar collector variant used only by ESCAPE_inference.py."""

    def __init__(self, *args, source_lidar_paths: Optional[Dict[str, str]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.source_lidar_paths: Dict[str, str] = dict(source_lidar_paths or {})

    def get_lidar_depth_image(self, lidar_path: str) -> Optional[np.ndarray]:
        """More tolerant depth read for runtime headless LiDAR prims."""
        try:
            depth_buffer = _call_lidar_interface(
                self.lidar_interface,
                ["get_depth_data", "getDepthData", "get_linear_depth_data", "getLinearDepthData"],
                lidar_path,
            )
            if depth_buffer is None:
                return None

            depth_buffer = np.asarray(depth_buffer, dtype=np.float32)
            if depth_buffer.size == 0:
                return None
            if depth_buffer.ndim == 2:
                return depth_buffer

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

            if num_rows <= 0 or num_cols <= 0:
                return depth_buffer.reshape(1, -1)
            if depth_buffer.size != num_rows * num_cols:
                return depth_buffer.reshape(1, -1)

            return depth_buffer.reshape(num_rows, num_cols)

        except Exception:
            return None

    def initialize(self, lidar_names: Optional[Iterable[str]] = None):
        """Initialize only LiDARs that are readable by the headless PhysX interface."""
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
            lidar_paths = {}
            for name in lidar_names:
                if name not in all_lidar_paths:
                    raise KeyError(
                        f"Requested lidar '{name}' not found. "
                        f"Available lidars: {list(all_lidar_paths.keys())}"
                    )
                lidar_paths[name] = all_lidar_paths[name]
        else:
            lidar_paths = all_lidar_paths

        lidar_infos: Dict[str, LidarStaticInfo] = {}

        for lidar_name, lidar_path in lidar_paths.items():
            source_lidar_path = self.source_lidar_paths.get(lidar_name, lidar_path)
            try:
                moving_link_path, moving_link_name, T_link_to_lidar = compute_static_transform_link_to_lidar(
                    lidar_path=source_lidar_path,
                    robot_root_keyword=self.robot_root_keyword,
                )
                sensor_attrs = self.get_lidar_static_sensor_attrs(lidar_path)
                depth_image = self.get_lidar_depth_image(lidar_path)
                if depth_image is not None and depth_image.size > 0:
                    sensor_attrs["num_rows"] = int(depth_image.shape[0])
                    sensor_attrs["num_cols"] = (
                        int(depth_image.shape[1]) if depth_image.ndim > 1 else int(depth_image.size)
                    )
            except Exception as exc:
                print(f"[HeadlessRadar] Skipping unreadable LiDAR {lidar_name}: {exc}")
                continue

            if int(sensor_attrs["num_rows"]) <= 0 or int(sensor_attrs["num_cols"]) <= 0:
                print(f"[HeadlessRadar] Skipping unregistered LiDAR {lidar_name}: {lidar_path}")
                continue

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

        if len(lidar_infos) == 0:
            raise RuntimeError(
                f"No valid headless PhysX LiDAR sensors found under env_prefix='{self.env_prefix}'. "
                f"Discovered prim names were {list(lidar_paths.keys())}."
            )

        self.lidar_infos = lidar_infos
        self.lidar_order = list(lidar_infos.keys())

    def sync_runtime_lidar_transforms(self, selected_lidars: Optional[Iterable[str]] = None) -> None:
        """Keep runtime LiDAR prims aligned with their source robot-mounted prims."""
        if not self.source_lidar_paths:
            return

        stage = omni.usd.get_context().get_stage()
        lidar_names: List[str]
        if selected_lidars is None:
            lidar_names = list(self.lidar_order)
        else:
            lidar_names = list(selected_lidars)

        for lidar_name in lidar_names:
            source_lidar_path = self.source_lidar_paths.get(lidar_name)
            info = self.lidar_infos.get(lidar_name)
            if source_lidar_path is None or info is None:
                continue
            runtime_prim = stage.GetPrimAtPath(info.lidar_path)
            if not runtime_prim.IsValid():
                continue
            xformable = UsdGeom.Xformable(runtime_prim)
            xformable.ClearXformOpOrder()
            xformable.AddTransformOp().Set(
                _np_column_convention_to_gf_matrix(
                    get_prim_world_transform_matrix(source_lidar_path)
                )
            )

    def collect_frame(
        self,
        include_depth: bool = False,
        merge: bool = True,
        selected_lidars: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        self.sync_runtime_lidar_transforms(selected_lidars=selected_lidars)
        return super().collect_frame(
            include_depth=include_depth,
            merge=merge,
            selected_lidars=selected_lidars,
        )


class HeadlessRaycastRadarPointCloudCollector:
    """Headless radar collector implemented with PhysX scene-query raycasts."""

    def __init__(
        self,
        robot,
        env_prefix: str = "/World/envs/env_0",
        env_id: int = 0,
        lidar_name_pattern: str = r"^Lidar\d+$",
        robot_root_keyword: str = "Robot",
        flip_local_y: bool = False,
        ignore_path_substrings: Optional[Iterable[str]] = None,
    ):
        self.robot = robot
        self.env_prefix = env_prefix
        self.env_id = env_id
        self.lidar_name_pattern = lidar_name_pattern
        self.robot_root_keyword = robot_root_keyword
        self.flip_local_y = flip_local_y
        self.ignore_path_substrings = tuple(
            ignore_path_substrings
            or (
                "/RuntimeLidars/",
                "/Visuals/",
                "/curobo",
                f"{self.env_prefix}/Robot",
            )
        )
        self.scene_query = omni.physx.get_physx_scene_query_interface()

        self.lidar_infos: Dict[str, LidarStaticInfo] = {}
        self.lidar_order: List[str] = []
        self.ray_dirs_by_lidar: Dict[str, np.ndarray] = {}
        self.usd_obstacle_triangles = np.zeros((0, 3, 3), dtype=np.float32)
        self.usd_obstacle_paths: List[str] = []

    def initialize(self, lidar_names: Optional[Iterable[str]] = None):
        all_lidar_paths = discover_lidars(
            env_prefix=self.env_prefix,
            lidar_name_pattern=self.lidar_name_pattern,
        )
        if len(all_lidar_paths) == 0:
            raise RuntimeError(
                f"No source lidars found under env_prefix='{self.env_prefix}' "
                f"with pattern='{self.lidar_name_pattern}'."
            )

        if lidar_names is not None:
            lidar_paths = {}
            for name in list(lidar_names):
                if name not in all_lidar_paths:
                    raise KeyError(
                        f"Requested lidar '{name}' not found. "
                        f"Available lidars: {list(all_lidar_paths.keys())}"
                    )
                lidar_paths[name] = all_lidar_paths[name]
        else:
            lidar_paths = all_lidar_paths

        lidar_infos: Dict[str, LidarStaticInfo] = {}
        ray_dirs_by_lidar: Dict[str, np.ndarray] = {}

        for lidar_name, lidar_path in lidar_paths.items():
            moving_link_path, moving_link_name, T_link_to_lidar = compute_static_transform_link_to_lidar(
                lidar_path=lidar_path,
                robot_root_keyword=self.robot_root_keyword,
            )
            attrs = _read_lidar_static_attrs_from_usd(lidar_path)
            ray_dirs = _build_ray_dirs_from_attrs(attrs)

            lidar_infos[lidar_name] = LidarStaticInfo(
                lidar_name=lidar_name,
                lidar_path=lidar_path,
                moving_link_name=moving_link_name,
                moving_link_path=moving_link_path,
                T_link_to_lidar=T_link_to_lidar.astype(np.float32),
                num_rows=int(attrs["num_rows"]),
                num_cols=int(attrs["num_cols"]),
                horizontal_fov_deg=attrs["horizontal_fov_deg"],
                vertical_fov_deg=attrs["vertical_fov_deg"],
                horizontal_resolution_deg=attrs["horizontal_resolution_deg"],
                vertical_resolution_deg=attrs["vertical_resolution_deg"],
                yaw_offset_deg=attrs["yaw_offset_deg"],
                min_range=attrs["min_range"],
                max_range=attrs["max_range"],
            )
            ray_dirs_by_lidar[lidar_name] = ray_dirs

        self.lidar_infos = lidar_infos
        self.lidar_order = list(lidar_infos.keys())
        self.ray_dirs_by_lidar = ray_dirs_by_lidar
        self.usd_obstacle_triangles, self.usd_obstacle_paths = _collect_usd_obstacle_triangles(self.env_prefix)
        print(
            f"[HeadlessRaycastRadar] USD mesh fallback: "
            f"{self.usd_obstacle_triangles.shape[0]} triangles from {len(self.usd_obstacle_paths)} prims"
        )

    def is_initialized(self) -> bool:
        return len(self.lidar_infos) > 0

    def get_lidar_names(self) -> List[str]:
        return list(self.lidar_order)

    def _should_ignore_hit(self, hit_path: str) -> bool:
        if not hit_path:
            return True
        return any(substr in hit_path for substr in self.ignore_path_substrings)

    def _raycast_depth(self, origin_w: np.ndarray, direction_w: np.ndarray, max_range: float) -> float:
        direction_w = np.asarray(direction_w, dtype=np.float32).reshape(3)
        norm = float(np.linalg.norm(direction_w))
        if norm <= 1e-8:
            return float(max_range)
        direction_w = direction_w / norm

        hits: List[Tuple[float, str]] = []

        def report_hit(hit):
            path = ""
            if isinstance(hit, dict):
                path = str(hit.get("rigidBody", hit.get("rigid_body", "")))
                distance = float(hit.get("distance", max_range))
                hits.append((distance, path))
                return True

            for attr_name in ("rigid_body", "rigidBody"):
                try:
                    path = getattr(hit, attr_name)
                    break
                except Exception:
                    pass
            try:
                distance = float(hit.distance)
            except Exception:
                distance = float(max_range)
            hits.append((distance, str(path)))
            return True

        try:
            self.scene_query.raycast_all(
                _as_gf_vec3(origin_w),
                _as_gf_vec3(direction_w),
                float(max_range),
                report_hit,
            )
        except TypeError:
            hit = self.scene_query.raycast_closest(
                tuple(float(x) for x in origin_w),
                tuple(float(x) for x in direction_w),
                float(max_range),
            )
            if hit.get("hit", False):
                path = str(hit.get("rigidBody", ""))
                if not self._should_ignore_hit(path):
                    return float(hit.get("distance", max_range))
            return _ray_triangle_nearest_distance(
                origin_w,
                direction_w,
                self.usd_obstacle_triangles,
                max_range,
            )

        for distance, path in sorted(hits, key=lambda item: item[0]):
            if distance <= 0.0:
                continue
            if self._should_ignore_hit(path):
                continue
            return float(distance)
        return _ray_triangle_nearest_distance(
            origin_w,
            direction_w,
            self.usd_obstacle_triangles,
            max_range,
        )

    def collect_frame(
        self,
        include_depth: bool = False,
        merge: bool = True,
        selected_lidars: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        if not self.is_initialized():
            raise RuntimeError("HeadlessRaycastRadarPointCloudCollector is not initialized.")

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
            ray_dirs_local = self.ray_dirs_by_lidar[lidar_name].reshape(-1, 3)

            T_link_to_world = get_link_world_transform_matrix(
                robot=self.robot,
                moving_link_name=info.moving_link_name,
                env_id=self.env_id,
                moving_link_path=info.moving_link_path,
            )
            T_lidar_to_world = (T_link_to_world @ info.T_link_to_lidar).astype(np.float32)
            origin_w = T_lidar_to_world[:3, 3]
            rot = T_lidar_to_world[:3, :3]

            directions_w = ray_dirs_local @ rot.T
            max_range = float(info.max_range or 4.0)
            min_range = float(info.min_range or 0.01)
            depth = np.array(
                [self._raycast_depth(origin_w, direction_w, max_range) for direction_w in directions_w],
                dtype=np.float32,
            )
            valid = np.isfinite(depth) & (depth >= min_range) & (depth < max_range)

            points_local_raw = ray_dirs_local * depth[:, None]
            points_local_raw[~valid] = 0.0
            points_local_fixed = points_local_raw.copy()
            if self.flip_local_y and points_local_fixed.size > 0:
                points_local_fixed[:, 1] *= -1.0

            points_world = transform_points_local_to_world(points_local_fixed, T_lidar_to_world)
            depth_image = depth.reshape(info.num_rows, info.num_cols) if include_depth else None

            per_lidar[lidar_name] = {
                "points_local_raw": points_local_raw.astype(np.float32),
                "points_local_fixed": points_local_fixed.astype(np.float32),
                "points_world": points_world.astype(np.float32),
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

            if merge and points_world.size > 0 and np.any(valid):
                merged_points_world_list.append(points_world[valid].astype(np.float32))
                merged_lidar_ids_list.append(
                    np.full((int(np.count_nonzero(valid)),), lidar_idx, dtype=np.int32)
                )
            if merge:
                merged_lidar_names.append(lidar_name)

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

    def collect_merged_world_points(self, selected_lidars: Optional[Iterable[str]] = None) -> np.ndarray:
        frame = self.collect_frame(include_depth=False, merge=True, selected_lidars=selected_lidars)
        return frame["merged_points_world"]

    def summarize_current_frame(self, selected_lidars: Optional[Iterable[str]] = None) -> Dict[str, Dict[str, Any]]:
        frame = self.collect_frame(include_depth=False, merge=False, selected_lidars=selected_lidars)
        summary = {}
        for lidar_name, data in frame["per_lidar"].items():
            summary[lidar_name] = {
                "world": pointcloud_stats(data["points_world"]),
                "lidar_path": data["lidar_path"],
                "moving_link_name": data["moving_link_name"],
                "moving_link_path": data["moving_link_path"],
            }
        return summary


class CpuRadarFallback:
    """Headless-safe radar substitute using USD obstacle samples and LiDAR poses."""

    def __init__(self, stage, robot, env_id: int, lidar_paths: Dict[str, str], obstacle_points: np.ndarray):
        self.stage = stage
        self.robot = robot
        self.env_id = env_id
        self.obstacle_points = np.asarray(obstacle_points, dtype=np.float32).reshape(-1, 3)
        self.lidar_infos = []
        for lidar_name, lidar_path in sorted(lidar_paths.items()):
            try:
                moving_link_path, moving_link_name, T_link_to_lidar = compute_static_transform_link_to_lidar(lidar_path)
            except Exception:
                continue
            self.lidar_infos.append((lidar_name, moving_link_path, moving_link_name, T_link_to_lidar))

    def collect_frame(
        self,
        include_depth: bool = True,
        merge: bool = False,
        selected_lidars: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        per_lidar = {}
        if self.obstacle_points.size == 0:
            return {"per_lidar": per_lidar}

        selected = None if selected_lidars is None else set(selected_lidars)
        for lidar_name, moving_link_path, moving_link_name, T_link_to_lidar in self.lidar_infos:
            if selected is not None and lidar_name not in selected:
                continue

            T_link_to_world = get_link_world_transform_matrix(
                self.robot,
                moving_link_name,
                env_id=self.env_id,
                moving_link_path=moving_link_path,
            )
            T_lidar_to_world = (T_link_to_world @ T_link_to_lidar).astype(np.float32)
            origin = T_lidar_to_world[:3, 3]
            rot = T_lidar_to_world[:3, :3]

            rel_world = self.obstacle_points - origin[None, :]
            points_local = rel_world @ rot
            depth = np.linalg.norm(points_local, axis=1).astype(np.float32)
            forward = points_local[:, 0]
            horizontal = np.degrees(np.arctan2(points_local[:, 1], np.maximum(forward, 1e-6)))
            vertical = np.degrees(np.arctan2(points_local[:, 2], np.maximum(forward, 1e-6)))
            mask = (
                (forward > 0.0)
                & (depth >= 0.01)
                & (depth <= 4.0)
                & (np.abs(horizontal) <= 45.0)
                & (np.abs(vertical) <= 45.0)
            )
            idx = np.flatnonzero(mask)
            if idx.size > 256:
                idx = idx[np.random.choice(idx.size, 256, replace=False)]

            pts_world = self.obstacle_points[idx].astype(np.float32)
            pts_local = points_local[idx].astype(np.float32)
            depth_valid = depth[idx].astype(np.float32)
            per_lidar[lidar_name] = {
                "points_world": pts_world,
                "points_local_fixed": pts_local,
                "depth_image": depth_valid.reshape(1, -1),
            }

        return {"per_lidar": per_lidar}
