"""Event terms for HDF5-based robot resets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_robot_only_usd(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
):
    """
    Reset robot state without switching the loaded USD scene.

    Each environment has its own fixed USD scene. Reset only updates robot joint positions.

    Args:
        env: The environment instance
        env_ids: Environment indices to reset
        asset_cfg: Robot asset configuration
    """
    scene_name = os.getenv("ESCAPE_SCENE_NAME", "scene_1")
    start_demo_id = int(os.getenv("ESCAPE_START_DEMO_ID", "0"))

    robot = env.scene[asset_cfg.name]

    hdf5_file = (
        Path(__file__).resolve().parents[7]
        / "assets"
        / "scenes"
        / "static_hdf5"
        / f"{scene_name}.hdf5"
    )

    # Load the initial robot state for each environment demo.
    with h5py.File(hdf5_file, "r") as f:
        for env_id in env_ids.tolist():
            # Environment ids map directly to demo ids.
            demo_id = start_demo_id + env_id
            demo_key = f"demo_{demo_id}"

            if demo_key not in f["data"]:
                print(f" Demo {demo_id} not found in HDF5, using default state")
                continue

            initial_joints = f["data"][demo_key]["obs"]["current_angles"][0]

            joint_pos = robot.data.default_joint_pos[env_id].clone()
            joint_pos[:7] = torch.tensor(
                initial_joints[:7], device=robot.device, dtype=torch.float32
            )
            joint_pos[7:9] = 0.04  # Gripper

            joint_vel = torch.zeros_like(robot.data.joint_vel[env_id])

            robot.write_joint_state_to_sim(
                joint_pos.unsqueeze(0),
                joint_vel.unsqueeze(0),
                env_ids=torch.tensor([env_id], device=robot.device)
            )

            if env_id == env_ids[0].item():
                print(f"[OK] Reset Env {env_id} to Demo {demo_id} (robot state only)")
