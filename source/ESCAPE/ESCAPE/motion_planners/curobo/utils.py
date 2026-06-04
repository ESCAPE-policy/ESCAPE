"""Coordinate frame conversion utilities for CuRobo motion planning.

CuRobo requires all coordinates in robot frame (robot base at origin).
This module provides conversion between world frame and robot frame.
"""

import numpy as np
from curobo.geom.types import WorldConfig, Cuboid


class WorldFrameConverter:
    """Convert coordinates between world frame and robot frame.
    
    Args:
        robot_world_pos: Robot base position in world frame [x, y, z]
        robot_world_quat: Robot base quaternion in world frame [w, x, y, z]
    """
    
    def __init__(self, robot_world_pos: np.ndarray, robot_world_quat: np.ndarray):
        self.robot_world_pos = robot_world_pos
        self.robot_world_quat = robot_world_quat
        
    def world_to_robot(self, world_pos: list) -> list:
        """Convert position from world frame to robot frame."""
        return [
            world_pos[0] - self.robot_world_pos[0],
            world_pos[1] - self.robot_world_pos[1],
            world_pos[2] - self.robot_world_pos[2],
        ]
    
    def robot_to_world(self, robot_pos: list) -> list:
        """Convert position from robot frame to world frame."""
        return [
            robot_pos[0] + self.robot_world_pos[0],
            robot_pos[1] + self.robot_world_pos[1],
            robot_pos[2] + self.robot_world_pos[2],
        ]
    
    def convert_world_config_to_robot(self, world_config: WorldConfig) -> WorldConfig:
        """Convert WorldConfig from world frame to robot frame.
        
        Args:
            world_config: WorldConfig with obstacles in world frame
            
        Returns:
            WorldConfig with obstacles in robot frame
        """
        robot_cuboids = []
        
        for cuboid in world_config.cuboid:
            world_pos = cuboid.pose[:3]
            quat = cuboid.pose[3:]
            robot_pos = self.world_to_robot(world_pos)
            
            robot_cuboid = Cuboid(
                name=cuboid.name,
                pose=robot_pos + quat,
                dims=cuboid.dims,
            )
            robot_cuboids.append(robot_cuboid)
        
        return WorldConfig(cuboid=robot_cuboids)