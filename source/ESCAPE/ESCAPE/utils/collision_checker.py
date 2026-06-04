"""Collision checker using joint tracking error + velocity."""

import torch
from typing import List, Tuple
from ESCAPE.robofin.samplers import TorchFrankaSampler
from ESCAPE.maniflow.mpiformer.geometry import TorchCuboids


class CollisionChecker:
    """Check collisions using joint tracking error + velocity.
    
    Collision signature: high tracking error AND low joint velocity.
    - Normal tracking lag: high error + high velocity (robot is catching up)
    - Collision/blocked:   high error + low velocity  (robot is stuck)
    """
    
    def __init__(self, collision_margin: float, device: str,
                 tracking_error_threshold: float = 0.1,
                 velocity_threshold: float = 1.0,
                 sustained_steps: int = 3):
        self.collision_margin = collision_margin
        self.device = device
        self.tracking_error_threshold = tracking_error_threshold
        self.velocity_threshold = velocity_threshold
        self.sustained_steps = sustained_steps
        self.fk_sampler = TorchFrankaSampler(
            num_robot_points=1024, num_eef_points=256,
            device=device, with_base_link=True, use_cache=True
        )
        
        self._target_joints = None
        self._consecutive_error_counts = {}
        self._prev_joints = {}
        self._last_velocity = {}
        self._last_stuck_joints = {}
    
    def set_target(self, target_pos: torch.Tensor, valid_env_ids: List[int]):
        self._target_joints = target_pos[:, :7].clone()
    
    def reset(self, valid_env_ids: List[int]):
        self._target_joints = None
        self._consecutive_error_counts = {env_id: 0 for env_id in valid_env_ids}
        self._prev_joints = {}
        self._last_velocity = {}
        self._last_stuck_joints = {}
    
    def check_tracking_error(self, valid_env_ids: List[int], robot,
                              env_active: torch.Tensor) -> torch.Tensor:
        n = len(valid_env_ids)
        collision_mask = torch.zeros(n, dtype=torch.bool, device=self.device)
        
        if self._target_joints is None:
            return collision_mask
        
        actual = robot.data.joint_pos[valid_env_ids, :7]
        target = self._target_joints[:n, :7]
        joint_errors = torch.abs(actual - target)
        dt = getattr(robot, '_sim_dt', 0.00833)
        if dt <= 0:
            dt = 0.00833
        
        for idx, env_id in enumerate(valid_env_ids):
            if not env_active[idx]:
                continue
            
            current = actual[idx]
            
            if env_id in self._prev_joints:
                joint_velocity = torch.abs(current - self._prev_joints[env_id]) / dt
            else:
                joint_velocity = torch.ones(7, device=self.device) * 999.0
            
            self._last_velocity[env_id] = joint_velocity.clone()
            self._prev_joints[env_id] = current.clone()
            
            high_error = joint_errors[idx] > self.tracking_error_threshold
            low_velocity = joint_velocity < self.velocity_threshold
            stuck_joints = high_error & low_velocity
            
            self._last_stuck_joints[env_id] = torch.where(stuck_joints)[0].tolist()
            
            if stuck_joints.any():
                self._consecutive_error_counts[env_id] = \
                    self._consecutive_error_counts.get(env_id, 0) + 1
            else:
                self._consecutive_error_counts[env_id] = 0
            
            if self._consecutive_error_counts[env_id] >= self.sustained_steps:
                collision_mask[idx] = True
        
        return collision_mask
    
    def debug_tracking(self, valid_env_ids: List[int], robot, env_active: torch.Tensor):
        if self._target_joints is None:
            return
        
        n = len(valid_env_ids)
        actual = robot.data.joint_pos[valid_env_ids, :7]
        target = self._target_joints[:n, :7]
        joint_errors = torch.abs(actual - target)
        
        for idx, env_id in enumerate(valid_env_ids):
            if not env_active[idx]:
                continue
            
            consecutive = self._consecutive_error_counts.get(env_id, 0)
            is_collision = consecutive >= self.sustained_steps
            
            worst_joint = joint_errors[idx].argmax().item()
            worst_err = joint_errors[idx, worst_joint].item()
            worst_vel = self._last_velocity[env_id][worst_joint].item() if env_id in self._last_velocity else -1.0
            stuck_ids = self._last_stuck_joints.get(env_id, [])
            
            status = "COLLISION" if is_collision else "OK"
            stuck_str = f", stuck_joints={stuck_ids}" if stuck_ids else ""
            
            print(f"    [Env {env_id}] err={worst_err:.4f}(j{worst_joint}), "
                  f"vel_j{worst_joint}={worst_vel:.4f}, "
                  f"consecutive={consecutive}/{self.sustained_steps}"
                  f"{stuck_str} {status}")
    
    def check_batch(self, joint_positions: torch.Tensor,
                    cuboid_centers: torch.Tensor,
                    cuboid_dims: torch.Tensor,
                    cuboid_quaternions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = joint_positions.shape[0]
        M = cuboid_centers.shape[1]
        
        robot_pc_with_idx = self.fk_sampler.sample(joint_positions[:, :7], 0.04)
        robot_pc = robot_pc_with_idx[..., :3]
        
        collision_mask = torch.zeros(B, dtype=torch.bool, device=self.device)
        near_miss_mask = torch.zeros(B, dtype=torch.bool, device=self.device)
        
        for obs_idx in range(M):
            valid_mask = cuboid_dims[:, obs_idx].abs().sum(dim=-1) > 1e-6
            if not valid_mask.any():
                continue
            
            cuboid = TorchCuboids(
                centers=cuboid_centers[:, obs_idx:obs_idx+1],
                dims=cuboid_dims[:, obs_idx:obs_idx+1],
                quaternions=cuboid_quaternions[:, obs_idx:obs_idx+1],
            )
            
            sdf = cuboid.sdf(robot_pc).squeeze(1)
            env_collision = (sdf < 0).any(dim=-1) & valid_mask
            collision_mask = collision_mask | env_collision
            
            env_near_miss = ((sdf >= 0) & (sdf < self.collision_margin)).any(dim=-1) & valid_mask
            near_miss_mask = near_miss_mask | (env_near_miss & ~env_collision)
        
        return collision_mask, near_miss_mask