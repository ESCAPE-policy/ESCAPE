"""USD geometry utilities: mesh conversion, point cloud sampling, obstacle data loading."""

import numpy as np
import torch
import trimesh
import h5py
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
from pxr import UsdGeom, Gf

from ESCAPE.scene_data.utils.lab_utils import decompose_scene_pcd_params_obs
from ESCAPE.robofin.samplers import TorchFrankaSampler


def _get_static_scene_hdf5_path(scene_name: str) -> Path:
    project_root = Path(__file__).resolve().parents[4]
    return project_root / "assets" / "scenes" / "static_hdf5" / f"{scene_name}.hdf5"


def prim_to_trimesh(prim) -> trimesh.Trimesh:
    """Convert USD prim to trimesh object."""
    tname = prim.GetTypeName()
    if tname == "Mesh":
        mesh = UsdGeom.Mesh(prim)
        vertices = np.array(mesh.GetPointsAttr().Get())
        face_vertex_counts = np.array(mesh.GetFaceVertexCountsAttr().Get())
        face_vertex_indices = np.array(mesh.GetFaceVertexIndicesAttr().Get())

        faces = []
        idx = 0
        for count in face_vertex_counts:
            if count == 3:
                faces.append(face_vertex_indices[idx:idx+3])
            elif count > 3:
                for i in range(1, count - 1):
                    faces.append([face_vertex_indices[idx],
                                  face_vertex_indices[idx + i],
                                  face_vertex_indices[idx + i + 1]])
            idx += count
        return trimesh.Trimesh(vertices=vertices, faces=np.array(faces), process=False)

    elif tname == "Cube":
        xform = UsdGeom.Xformable(prim)
        world_transform = xform.ComputeLocalToWorldTransform(0)
        scale = Gf.Vec3d(*(v.GetLength() for v in world_transform.ExtractRotationMatrix()))
        size_attr = prim.GetAttribute("size")
        size = size_attr.Get() if size_attr and size_attr.HasAuthoredValue() else 2.0
        box = trimesh.creation.box(extents=(size, size, size))
        box.apply_scale(np.array(scale))
        return box
    else:
        raise ValueError(f"Unsupported prim type: {tname}")


def sample_pointcloud_from_mesh_local(prim, num_points: int) -> np.ndarray:
    """Sample point cloud from mesh in local coordinates."""
    mesh = UsdGeom.Mesh(prim)
    points = np.array(mesh.GetPointsAttr().Get())
    face_vertex_counts = np.array(mesh.GetFaceVertexCountsAttr().Get())
    face_vertex_indices = np.array(mesh.GetFaceVertexIndicesAttr().Get())

    if points.size == 0:
        raise ValueError(f"Empty mesh at {prim.GetPath()}")

    faces = []
    idx = 0
    for count in face_vertex_counts:
        if count == 3:
            faces.append(face_vertex_indices[idx:idx+3])
        elif count > 3:
            for i in range(1, count - 1):
                faces.append([face_vertex_indices[idx],
                              face_vertex_indices[idx + i],
                              face_vertex_indices[idx + i + 1]])
        idx += count

    mesh_tm = trimesh.Trimesh(vertices=points, faces=np.array(faces), process=False)
    return mesh_tm.sample(num_points)


def sample_pointcloud_from_cube_local(prim, num_points: int) -> np.ndarray:
    """Sample point cloud from cube in local coordinates."""
    size_attr = prim.GetAttribute("size")
    size = size_attr.Get() if size_attr and size_attr.HasAuthoredValue() else 2.0

    xform = UsdGeom.Xformable(prim)
    world_transform = xform.ComputeLocalToWorldTransform(0)
    scale = Gf.Vec3d(*(v.GetLength() for v in world_transform.ExtractRotationMatrix()))

    mesh_tm = trimesh.creation.box(extents=(size, size, size))
    points = mesh_tm.sample(num_points)
    return points * np.array(scale)


def compute_obstacle_surface_areas(stage, env_id: int, demo_id: int):
    """Compute surface areas for all obstacles in current demo.
    
    Returns:
        obstacle_areas: np.ndarray of surface areas
        obstacle_prims: list of USD prims
        obstacle_names: list of obstacle names
    """
    prefix = f"obstacle_{demo_id}_"
    obstacle_areas = []
    obstacle_prims = []
    obstacle_names = []

    obstacles_root = f"/World/envs/env_{env_id}/obstacles/Obstacles"
    obstacles_prim = stage.GetPrimAtPath(obstacles_root)

    if not obstacles_prim.IsValid():
        obstacles_root = f"/World/envs/env_{env_id}/obstacles"
        obstacles_prim = stage.GetPrimAtPath(obstacles_root)

    if not obstacles_prim.IsValid():
        print(f"    [DEBUG] Could not find obstacles root")
        return np.array([]), [], []

    for child in obstacles_prim.GetChildren():
        child_name = child.GetName()

        if child_name == "Obstacles":
            for obstacle_prim in child.GetChildren():
                obstacle_name = obstacle_prim.GetName()
                if obstacle_name.startswith(prefix):
                    mesh_prim = stage.GetPrimAtPath(f"{obstacle_prim.GetPath()}/geometry/mesh")
                    if mesh_prim.IsValid():
                        area = prim_to_trimesh(mesh_prim).area
                        obstacle_areas.append(area)
                        obstacle_prims.append(mesh_prim)
                        obstacle_names.append(obstacle_name)

        elif child_name.startswith(prefix):
            mesh_prim = stage.GetPrimAtPath(f"{child.GetPath()}/geometry/mesh")
            if mesh_prim.IsValid():
                area = prim_to_trimesh(mesh_prim).area
                obstacle_areas.append(area)
                obstacle_prims.append(mesh_prim)
                obstacle_names.append(child_name)

    print(f"    [DEBUG] Found {len(obstacle_areas)} obstacles under {obstacles_root}")
    return np.array(obstacle_areas), obstacle_prims, obstacle_names


def collect_obstacle_pointclouds(stage, env_id: int,
                                 obstacle_areas: np.ndarray,
                                 obstacle_prims: list,
                                 obstacle_names: list,
                                 total_points: int) -> np.ndarray:
    """Collect point clouds from all obstacles in world coordinates."""
    if len(obstacle_areas) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    total_area = obstacle_areas.sum()
    points_per_obs = (obstacle_areas / total_area * total_points).astype(int)
    remaining = total_points - points_per_obs.sum()
    if remaining > 0:
        largest = np.argsort(obstacle_areas)[::-1][:remaining]
        points_per_obs[largest] += 1

    pcs = []
    for num_pts, mesh_prim, name in zip(points_per_obs, obstacle_prims, obstacle_names):
        if num_pts == 0:
            continue

        prim_type = mesh_prim.GetTypeName()
        if prim_type == "Mesh":
            pts_local = sample_pointcloud_from_mesh_local(mesh_prim, num_pts)
        else:
            pts_local = sample_pointcloud_from_cube_local(mesh_prim, num_pts)

        obstacle_prim_path = mesh_prim.GetPath().GetParentPath().GetParentPath()
        obstacle_prim = stage.GetPrimAtPath(obstacle_prim_path)
        if not obstacle_prim.IsValid():
            continue

        xform = UsdGeom.Xformable(obstacle_prim)
        mat = np.array(xform.ComputeLocalToWorldTransform(0)).T

        pts_h = np.concatenate([pts_local, np.ones((pts_local.shape[0], 1))], axis=1)
        pts_world = (mat @ pts_h.T).T[:, :3]
        pcs.append(pts_world)

    return np.concatenate(pcs, axis=0).astype(np.float32) if pcs else np.zeros((0, 3), dtype=np.float32)


def collect_robot_pointcloud_from_fk(joint_positions: torch.Tensor,
                                     fk_sampler: TorchFrankaSampler,
                                     num_points: int = 1024) -> np.ndarray:
    """Sample robot point cloud using FK sampler."""
    if joint_positions.dim() == 1:
        joint_positions = joint_positions.unsqueeze(0)

    robot_pc_with_idx = fk_sampler.sample(joint_positions[:, :7], 0.04, num_points=num_points)
    return robot_pc_with_idx[0, :, :3].cpu().numpy()


@dataclass
class RobotUsdPointCloudCache:
    """Cached robot surface samples grouped by articulation body."""

    robot_root_path: str
    body_name_to_index: Dict[str, int]
    points_by_body: Dict[str, np.ndarray]
    geometry_paths_by_body: Dict[str, List[str]]

    @property
    def total_points(self) -> int:
        return int(sum(points.shape[0] for points in self.points_by_body.values()))

    @property
    def total_links(self) -> int:
        return int(sum(1 for points in self.points_by_body.values() if len(points) > 0))

    @property
    def total_geometries(self) -> int:
        return int(sum(len(paths) for paths in self.geometry_paths_by_body.values()))


def _prim_to_trimesh_local(prim) -> trimesh.Trimesh:
    """Convert a geometry prim to trimesh in the prim's local frame."""
    tname = prim.GetTypeName()
    if tname == "Mesh":
        mesh = UsdGeom.Mesh(prim)
        vertices = np.array(mesh.GetPointsAttr().Get())
        face_vertex_counts = np.array(mesh.GetFaceVertexCountsAttr().Get())
        face_vertex_indices = np.array(mesh.GetFaceVertexIndicesAttr().Get())

        faces = []
        idx = 0
        for count in face_vertex_counts:
            if count == 3:
                faces.append(face_vertex_indices[idx:idx+3])
            elif count > 3:
                for i in range(1, count - 1):
                    faces.append([face_vertex_indices[idx],
                                  face_vertex_indices[idx + i],
                                  face_vertex_indices[idx + i + 1]])
            idx += count
        return trimesh.Trimesh(vertices=vertices, faces=np.array(faces), process=False)

    if tname == "Cube":
        size_attr = prim.GetAttribute("size")
        size = size_attr.Get() if size_attr and size_attr.HasAuthoredValue() else 2.0
        return trimesh.creation.box(extents=(size, size, size))

    raise ValueError(f"Unsupported prim type for robot sampling: {tname}")


def _get_local_transform_matrix_from_prim(prim) -> np.ndarray:
    """Read one prim's local transform matrix using the repo's column convention."""
    xformable = UsdGeom.Xformable(prim)
    result = xformable.GetLocalTransformation()
    gf_mat = result[0] if isinstance(result, tuple) else result
    mat = np.array([[gf_mat[i][j] for j in range(4)] for i in range(4)], dtype=np.float32)
    return mat.T


def _compute_relative_transform_matrix(parent_prim, child_prim) -> np.ndarray:
    """Compute the static transform from parent prim frame to child prim frame."""
    parent_path = parent_prim.GetPath().pathString
    child_path = child_prim.GetPath().pathString
    if parent_path == child_path:
        return np.eye(4, dtype=np.float32)

    prefix = parent_path.rstrip("/") + "/"
    if not child_path.startswith(prefix):
        raise ValueError(f"{child_path} is not a descendant of {parent_path}")

    relative_parts = child_path[len(prefix):].split("/")
    stage = child_prim.GetStage()
    current_path = parent_path
    transform = np.eye(4, dtype=np.float32)
    for child_name in relative_parts:
        current_path = current_path + "/" + child_name
        current_prim = stage.GetPrimAtPath(current_path)
        transform = transform @ _get_local_transform_matrix_from_prim(current_prim)
    return transform


def _iter_descendant_geometry_prims(root_prim) -> List:
    """Collect supported geometry prims below a link prim."""
    supported = {"Mesh", "Cube"}
    geometry_prims = []
    stack = list(root_prim.GetChildren())
    while stack:
        prim = stack.pop()
        stack.extend(list(prim.GetChildren()))
        if prim.GetTypeName() not in supported:
            continue
        geometry_prims.append(prim)

    return geometry_prims


def _quat_wxyz_to_rotmat(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert wxyz quaternion to a 3x3 rotation matrix."""
    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float64)
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)

    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z

    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float32,
    )


def build_robot_pointcloud_cache_from_usd(
    stage,
    robot,
    env_id: int = 0,
    num_points: int = 1024,
    robot_root_path: str | None = None,
    exclude_body_names: List[str] | None = None,
    body_sampling_weights: Dict[str, float] | None = None,
) -> RobotUsdPointCloudCache:
    """Build a cached robot surface sampler from the currently loaded robot USD."""
    if robot_root_path is None:
        robot_root_path = f"/World/envs/env_{env_id}/Robot"

    robot_root_prim = stage.GetPrimAtPath(robot_root_path)
    if not robot_root_prim.IsValid():
        raise ValueError(f"Invalid robot root path: {robot_root_path}")

    body_names = list(getattr(robot, "body_names", []))
    body_name_to_index = {name: idx for idx, name in enumerate(body_names)}
    excluded = set(exclude_body_names or [])
    body_weights = dict(body_sampling_weights or {})
    geometry_entries = []

    for body_name in body_names:
        if body_name in excluded:
            continue

        link_prim = stage.GetPrimAtPath(f"{robot_root_path}/{body_name}")
        if not link_prim.IsValid():
            continue

        for geom_prim in _iter_descendant_geometry_prims(link_prim):
            try:
                mesh_local = _prim_to_trimesh_local(geom_prim)
                transform_link_to_geom = _compute_relative_transform_matrix(link_prim, geom_prim)
                mesh_in_link = mesh_local.copy()
                mesh_in_link.apply_transform(transform_link_to_geom)
                area = float(mesh_in_link.area)
            except Exception:
                continue

            if area <= 0.0:
                continue

            sampling_weight = max(float(body_weights.get(body_name, 1.0)), 0.0)
            if sampling_weight <= 0.0:
                continue

            geometry_entries.append(
                {
                    "body_name": body_name,
                    "path": geom_prim.GetPath().pathString,
                    "mesh_local": mesh_local,
                    "transform_link_to_geom": transform_link_to_geom,
                    "area": area,
                    "weighted_area": area * sampling_weight,
                }
            )

    if not geometry_entries:
        return RobotUsdPointCloudCache(
            robot_root_path=robot_root_path,
            body_name_to_index=body_name_to_index,
            points_by_body={},
            geometry_paths_by_body={},
        )

    areas = np.array([entry["weighted_area"] for entry in geometry_entries], dtype=np.float64)
    total_area = float(np.clip(areas.sum(), 1e-8, None))
    points_per_geom = np.floor(areas / total_area * int(num_points)).astype(int)
    remaining = int(num_points) - int(points_per_geom.sum())
    if remaining > 0:
        largest_indices = np.argsort(areas)[::-1][:remaining]
        points_per_geom[largest_indices] += 1

    points_by_body_raw: Dict[str, List[np.ndarray]] = defaultdict(list)
    geometry_paths_by_body: Dict[str, List[str]] = defaultdict(list)
    for entry, geom_points in zip(geometry_entries, points_per_geom):
        if geom_points <= 0:
            continue

        sampled_local = entry["mesh_local"].sample(int(geom_points)).astype(np.float32)
        sampled_local_h = np.concatenate(
            [sampled_local, np.ones((sampled_local.shape[0], 1), dtype=np.float32)], axis=1
        )
        sampled_in_link = (
            entry["transform_link_to_geom"] @ sampled_local_h.T
        ).T[:, :3].astype(np.float32)

        points_by_body_raw[entry["body_name"]].append(sampled_in_link)
        geometry_paths_by_body[entry["body_name"]].append(entry["path"])

    points_by_body = {
        body_name: np.concatenate(points_list, axis=0).astype(np.float32)
        for body_name, points_list in points_by_body_raw.items()
        if points_list
    }

    return RobotUsdPointCloudCache(
        robot_root_path=robot_root_path,
        body_name_to_index=body_name_to_index,
        points_by_body=points_by_body,
        geometry_paths_by_body={k: list(v) for k, v in geometry_paths_by_body.items()},
    )


def collect_robot_pointcloud_from_usd(
    robot,
    env_id: int,
    cache: RobotUsdPointCloudCache,
    num_points: int | None = None,
) -> np.ndarray:
    """Transform cached per-link USD samples into the current world frame."""
    if cache is None or not cache.points_by_body:
        target = int(num_points) if num_points is not None else 0
        return np.zeros((target, 3), dtype=np.float32)

    pcs = []
    for body_name, points_local in cache.points_by_body.items():
        body_idx = cache.body_name_to_index.get(body_name, None)
        if body_idx is None or points_local.size == 0:
            continue

        body_state = robot.data.body_state_w[env_id, body_idx, 0:7].clone().cpu().numpy()
        pos = body_state[:3].astype(np.float32)
        quat_wxyz = body_state[3:7].astype(np.float32)
        rot = _quat_wxyz_to_rotmat(quat_wxyz)
        pcs.append((points_local @ rot.T + pos).astype(np.float32))

    if pcs:
        robot_pc = np.concatenate(pcs, axis=0).astype(np.float32)
    else:
        robot_pc = np.zeros((0, 3), dtype=np.float32)

    if num_points is None or robot_pc.shape[0] == int(num_points):
        return robot_pc

    target_points = int(num_points)
    if robot_pc.shape[0] == 0:
        return np.zeros((target_points, 3), dtype=np.float32)
    if robot_pc.shape[0] > target_points:
        indices = np.random.choice(robot_pc.shape[0], target_points, replace=False)
        return robot_pc[indices].astype(np.float32)

    indices = np.random.choice(robot_pc.shape[0], target_points, replace=True)
    return robot_pc[indices].astype(np.float32)


def load_cuboid_info_from_hdf5(scene_name: str, demo_id: int) -> dict:
    """Load cuboid geometry from HDF5."""
    hdf5_path = _get_static_scene_hdf5_path(scene_name)

    with h5py.File(hdf5_path, 'r') as f:
        pcd_params = f["data"][f"demo_{demo_id}"]["obs"]["compute_pcd_params"][0]
        dims, centers, quats, *_ = decompose_scene_pcd_params_obs(pcd_params[1:])

        return {
            'cuboid_centers': np.array(centers, dtype=np.float32),
            'cuboid_dims': np.array(dims, dtype=np.float32),
            'cuboid_quaternions': np.array(quats, dtype=np.float32),
        }


def load_obstacle_data_from_hdf5(scene_name: str, demo_id: int,
                                  device: str) -> Dict[str, torch.Tensor]:
    """Load obstacle cuboid data from scene HDF5."""
    hdf5_path = _get_static_scene_hdf5_path(scene_name)

    with h5py.File(hdf5_path, 'r') as f:
        pcd_params = f["data"][f"demo_{demo_id}"]["obs"]["compute_pcd_params"][0]
        dims, centers, quats, *_ = decompose_scene_pcd_params_obs(pcd_params[1:])

        quats = np.array(quats, dtype=np.float32)[:, [3, 0, 1, 2]]  # xyzw -> wxyz

        return {
            'centers': torch.tensor(centers, dtype=torch.float32, device=device),
            'dims': torch.tensor(dims, dtype=torch.float32, device=device),
            'quaternions': torch.tensor(quats, dtype=torch.float32, device=device),
        }
