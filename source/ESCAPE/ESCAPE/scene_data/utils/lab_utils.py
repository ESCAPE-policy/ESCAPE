"""
Scene configuration parsing utilities that do not depend on isaacgym.
"""

import numpy as np

def decompose_scene_pcd_params_obs(scene_pcd_params):
    """
    Decompose the pcd params observation.
    """
    M = int(scene_pcd_params[0])

    cuboid_params = scene_pcd_params[1 : 1 + 10 * M]
    cuboid_dims = cuboid_params[: 3 * M].reshape(-1, 3)
    cuboid_centers = cuboid_params[3 * M : 6 * M].reshape(-1, 3)
    cuboid_quats = cuboid_params[6 * M : 10 * M].reshape(-1, 4)

    cylinder_params = scene_pcd_params[1 + 10 * M : 1 + 10 * M + 9 * M]
    cylinder_radii = cylinder_params[: 1 * M].reshape(-1)
    cylinder_heights = cylinder_params[1 * M : 2 * M].reshape(-1)
    cylinder_centers = cylinder_params[2 * M : 5 * M].reshape(-1, 3)
    cylinder_quats = cylinder_params[5 * M : 9 * M].reshape(-1, 4)

    sphere_params = scene_pcd_params[1 + 10 * M + 9 * M : 1 + 10 * M + 9 * M + 4 * M]
    sphere_centers = sphere_params[: 3 * M].reshape(-1, 3)
    sphere_radii = sphere_params[3 * M : 4 * M].reshape(-1)

    mesh_params = scene_pcd_params[1 + 10 * M + 9 * M + 4 * M :]
    mesh_positions = mesh_params[: 3 * M].reshape(-1, 3)
    mesh_scales = mesh_params[3 * M : 4 * M].reshape(-1)
    mesh_quaternions = mesh_params[4 * M : 8 * M].reshape(-1, 4)
    obj_ids = mesh_params[8 * M : 9 * M].reshape(-1)
    mesh_ids = mesh_params[9 * M : 10 * M].reshape(-1)

    return (
        np.array(cuboid_dims).astype(np.float32),
        np.array(cuboid_centers).astype(np.float32),
        np.array(cuboid_quats).astype(np.float32),
        np.array(cylinder_radii).astype(np.float32),
        np.array(cylinder_heights).astype(np.float32),
        np.array(cylinder_centers).astype(np.float32),
        np.array(cylinder_quats).astype(np.float32),
        np.array(sphere_centers).astype(np.float32),
        np.array(sphere_radii).astype(np.float32),
        np.array(
            mesh_positions,
        ).astype(np.float32),
        np.array(
            mesh_scales,
        ).astype(np.float32),
        np.array(
            mesh_quaternions,
        ).astype(np.float32),
        np.array(
            obj_ids,
        ).astype(np.float32),
        np.array(
            mesh_ids,
        ).astype(np.float32),
    )