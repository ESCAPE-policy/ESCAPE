# Copyright (c) 2024-2025, ESCAPE Project
# SPDX-License-Identifier: Apache-2.0

"""Franka HDF5 scene task configuration."""

import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Franka-HDF5-USD-all-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaHDF5Env_USD_ALL_Cfg",
    },
    disable_env_checker=True,
)
