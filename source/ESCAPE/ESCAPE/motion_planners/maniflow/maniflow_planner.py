import torch
import numpy as np
from typing import Any, Dict, Optional
from pathlib import Path

from isaaclab.assets import Articulation
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from ESCAPE.motion_planners.motion_planner_base import MotionPlannerBase

# Import ManiFlow policy
import sys
import pathlib
ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Keep Hydra checkpoint instantiation compatible with the radar-pointcloud policy.
from maniflow.policy.maniflow_pointcloud_policy_goal_radar_pointcloud import ManiFlowTransformerPointcloudPolicyForce
import dill


def _to_device_tensor(x: Any, device: torch.device) -> torch.Tensor:
    """Convert numpy/tensor/list input to float tensor on target device."""
    if torch.is_tensor(x):
        return x.to(device=device, dtype=torch.float32)
    return torch.as_tensor(x, dtype=torch.float32, device=device)


class ManiFlowPlanner(MotionPlannerBase):
    """ManiFlow-based motion planner using trained checkpoint.
    
    This planner uses a trained ManiFlow model to generate collision-free trajectories
    from point cloud observations and target end-effector poses.
    
    Example:
        >>> from ESCAPE.motion_planners.maniflow.maniflow_planner import ManiFlowPlanner
        >>> planner = ManiFlowPlanner(env, robot, checkpoint_path="path/to/checkpoint.ckpt")
        >>> success = planner.update_world_and_plan_motion(target_pose, point_cloud)
        >>> if success:
        >>>     while planner.has_next_waypoint():
        >>>         joint_target = planner.get_next_waypoint_joint_pos()
        >>>         # Execute joint_target
    """

    def __init__(
        self,
        env: ManagerBasedEnv,
        robot: Articulation,
        checkpoint_path: str,
        env_id: int = 0,
        debug: bool = False,
        device: str = "cuda:0",
        num_inference_steps: int = 10,
        horizon: int = 16,
        n_action_steps: int = 16,
        n_obs_steps: int = 2,
        **kwargs
    ):
        """Initialize ManiFlow planner.
        
        Args:
            env: The environment instance
            robot: Robot articulation
            checkpoint_path: Path to trained ManiFlow checkpoint
            env_id: Environment ID
            debug: Enable debug output
            device: Device to run inference on
            num_inference_steps: Number of diffusion inference steps
            horizon: Action horizon
            n_action_steps: Number of action steps to execute
            n_obs_steps: Number of observation steps
        """
        super().__init__(env, robot, env_id, debug, **kwargs)
        
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path)
        
        # Planning parameters
        self.num_inference_steps = num_inference_steps
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        
        # Observation history buffer
        self.obs_history = {
            'point_cloud': [],
            'agent_pos': [],
        }
        self.max_history_len = n_obs_steps
        
        # Load policy
        self._load_policy()
        
        # Planning state
        self.current_plan = None
        self.waypoint_idx = 0

        if self.debug:
            print(f"[ManiFlowPlanner] Initialized for env {env_id}")
            print(f"  Checkpoint: {self.checkpoint_path}")
            print(f"  Device: {self.device}")
            print(f"  Inference steps: {self.num_inference_steps}")

    def _load_policy(self):
        """Load trained ManiFlow policy from checkpoint."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        # Load checkpoint
        payload = torch.load(self.checkpoint_path, pickle_module=dill, map_location='cpu')
        cfg = payload['cfg']
        
        # Create the policy instance from the checkpoint config.
        import hydra
        self.policy = hydra.utils.instantiate(cfg.policy)
        
        # Load EMA model if available
        if cfg.training.use_ema and 'ema_model' in payload['state_dicts']:
            self.policy.load_state_dict(payload['state_dicts']['ema_model'])
            if self.debug:
                print("[ManiFlowPlanner] Loaded EMA model")
        else:
            self.policy.load_state_dict(payload['state_dicts']['model'])
            if self.debug:
                print("[ManiFlowPlanner] Loaded base model")
        
        # Set to eval mode and move to device
        self.policy.eval()
        self.policy.to(self.device)
        
        # Override inference steps if specified
        self.policy.num_inference_steps = self.num_inference_steps

    def update_world_and_plan_motion(
        self,
        target_pose: Optional[torch.Tensor] = None,
        target_joint_pos: Optional[torch.Tensor] = None,
        point_cloud: Optional[np.ndarray] = None,
        current_joint_pos: Optional[np.ndarray] = None,
        **kwargs
    ) -> bool:
        """Plan motion to target using point cloud observation.
        
        Args:
            target_pose: Target end-effector pose (4x4 matrix) or None
            target_joint_pos: Target joint positions (7,) or None
            point_cloud: Point cloud observation (N, 6) [xyz + rgb/normal]
            current_joint_pos: Current joint positions (7,)
            
        Returns:
            bool: True if planning succeeded
        """
        # Prepare observation
        if point_cloud is None or current_joint_pos is None:
            raise ValueError("point_cloud and current_joint_pos must be provided for ManiFlowPlanner.")

        obs_dict = self._prepare_observation(
            point_cloud=point_cloud,
            current_joint_pos=current_joint_pos,
            target_pose=target_pose,
            target_joint_pos=target_joint_pos,
        )
        if self.debug:
            print(f"[DEBUG] obs_dict keys: {obs_dict.keys()}")
            for key, value in obs_dict.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")
                    print(f"    range: [{value.min():.3f}, {value.max():.3f}]")
            
        # Run inference
        with torch.no_grad():
            result = self.policy.predict_action(obs_dict)

        # Prefer the already windowed action sequence when available.
        self.current_plan = result.get("action", result.get("action_pred"))  # (1, n_action_steps|horizon, 7)
        if self.current_plan is None:
            raise KeyError("Policy output must contain either 'action' or 'action_pred'.")

        self.waypoint_idx = 0
            
        if self.debug:
            print(f"[ManiFlowPlanner] Planning succeeded")
            print(f"  Plan shape: {self.current_plan.shape}")
            print(f"  Horizon: {self.horizon}")
            
        return True
            


    def _prepare_observation(
        self,
        point_cloud: Any,
        current_joint_pos: Any,
        target_pose: Optional[torch.Tensor] = None,
        target_joint_pos: Optional[Any] = None,
    ) -> dict:
        """Prepare observation with real temporal history"""
        
        # Convert to tensors and move to device
        point_cloud = _to_device_tensor(point_cloud, self.device)
        current_joint_pos = _to_device_tensor(current_joint_pos, self.device)
        
        # Update history buffer
        self.obs_history['point_cloud'].append(point_cloud)
        self.obs_history['agent_pos'].append(current_joint_pos)
        
        # Maintain sliding window
        if len(self.obs_history['point_cloud']) > self.max_history_len:
            self.obs_history['point_cloud'].pop(0)
            self.obs_history['agent_pos'].pop(0)
        
        # Pad if insufficient history (cold start)
        while len(self.obs_history['point_cloud']) < self.n_obs_steps:
            self.obs_history['point_cloud'].insert(0, point_cloud)
            self.obs_history['agent_pos'].insert(0, current_joint_pos)
        
        # Stack into sequence
        point_cloud_seq = torch.stack(self.obs_history['point_cloud'], dim=0).unsqueeze(0)
        agent_pos_seq = torch.stack(self.obs_history['agent_pos'], dim=0).unsqueeze(0)
        
        # Prepare goal
        if target_joint_pos is not None:
            goal = _to_device_tensor(target_joint_pos, self.device)
        else:
            goal = current_joint_pos
        
        goal_batch = goal.unsqueeze(0)
        
        obs_dict = {
            'point_cloud': point_cloud_seq,  # (1, n_obs_steps, N, 6)
            'agent_pos': agent_pos_seq,      # (1, n_obs_steps, 7)
            'goal': goal_batch,              # (1, 7)
        }
        
        return obs_dict

    def has_next_waypoint(self) -> bool:
        """Check if there are more waypoints in current plan."""
        if self.current_plan is None:
            return False
        max_steps = min(self.n_action_steps, self.current_plan.shape[1])
        return self.waypoint_idx < max_steps

    def get_next_waypoint_joint_pos(self) -> torch.Tensor:
        """Get next waypoint's joint positions.
        
        Returns:
            torch.Tensor: Joint positions (7,) for the next waypoint
        """
        if not self.has_next_waypoint():
            raise RuntimeError("No more waypoints in plan")
        
        # Get joint positions from plan
        joint_pos = self.current_plan[0, self.waypoint_idx, :7]  # (7,)
        self.waypoint_idx += 1
        
        return joint_pos

    def get_next_waypoint_ee_pose(self) -> torch.Tensor:
        """Get next waypoint's end-effector pose.
        
        Note: This requires forward kinematics. For now, return joint positions.
        """
        return self.get_next_waypoint_joint_pos()

    def reset_plan(self) -> None:
        """Reset plan and observation history"""
        self.current_plan = None
        self.waypoint_idx = 0
        self.obs_history = {'point_cloud': [], 'agent_pos': []}

    def get_planner_info(self) -> dict:
        """Get planner information."""
        info = super().get_planner_info()
        info.update({
            "checkpoint": str(self.checkpoint_path),
            "num_inference_steps": self.num_inference_steps,
            "horizon": self.horizon,
            "n_action_steps": self.n_action_steps,
        })
        return info
