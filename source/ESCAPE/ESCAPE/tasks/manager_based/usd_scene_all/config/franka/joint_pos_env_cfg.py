"""
Joint Position Environment Configuration for Franka Robot with USD Scene Support
Loads ALL demo scenes at initialization (1 demo = 1 environment)
"""

from pathlib import Path
import h5py
import numpy as np
import os
from termcolor import cprint

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    TerminationTermCfg as DoneTerm,
    SceneEntityCfg,
)
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import ActionTermCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
import isaaclab_tasks.manager_based.manipulation.lift.mdp as mdp
from isaaclab.scene import InteractiveSceneCfg

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
from ESCAPE.tasks.manager_based.usd_scene_all.mdp import hdf5_events


def _check_radar_enabled() -> bool:
    """Check if radar is enabled from environment variable"""
    env_val = os.getenv("ESCAPE_ENABLE_RADAR", "false")
    enabled = env_val.lower() in ("true", "1", "yes")

    print(f"[joint_pos_env_cfg_usd] ESCAPE_ENABLE_RADAR='{env_val}' -> {'ENABLED' if enabled else 'DISABLED'}")
    return enabled


def _get_usd_scene_path(scene_name: str, demo_id: int) -> str:
    """Get USD scene file path"""
    usd_filename = f"{scene_name}_demo_{demo_id}.usd"
    project_root = Path(__file__).resolve().parents[8]
    return str(project_root / "assets" / "scenes" / "usd" / usd_filename)


def _get_static_scene_hdf5_path(scene_name: str) -> str:
    """Get the static scene HDF5 path for robot reset state."""
    project_root = Path(__file__).resolve().parents[8]
    return str(project_root / "assets" / "scenes" / "static_hdf5" / f"{scene_name}.hdf5")


def _get_obstacle_wood_mdl_path() -> str:
    """Get the repository-local Oak Planks MDL material path for obstacles."""
    project_root = Path(__file__).resolve().parents[8]
    return str(project_root / "assets" / "textures" / "Oak_Planks.mdl")


@configclass
class FrankaUSDSceneCfg_ALL(InteractiveSceneCfg):
    cprint(f"\n[FrankaUSDSceneCfg_ALL] Initializing USD Scene Configuration...", "red", attrs=['bold'])
    """
    Scene configuration with USD-based obstacle loading
    Environment Variables:
        ESCAPE_ENABLE_RADAR: Enable radar sensor
        ESCAPE_START_DEMO_ID: Starting demo index
        ESCAPE_SCENE_NAME: Scene name
        ESCAPE_NUM_DEMOS: Number of demos to load
    """
    # different scene settings
    replicate_physics: bool = False

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # Lighting
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DistantLightCfg(
            intensity=3000.0,
            color=(0.75, 0.75, 0.75),
        ),
    )

    front_light = AssetBaseCfg(
        prim_path="/World/FrontLight",
        spawn=sim_utils.DistantLightCfg(
            intensity=1500.0,
            color=(1.0, 0.95, 0.85),  # warm white light
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(2.0, 0.0, 2.0),
            rot=(0.924, 0.0, -0.383, 0.0),  # roughly 45 degrees downward
        ),
    )

    def __post_init__(self):
        ENABLE_RADAR = _check_radar_enabled()

        # Load configuration from environment
        start_demo_id = int(os.getenv("ESCAPE_START_DEMO_ID", os.getenv("ESCAPE_DEMO_ID", "0")))
        scene_name = os.getenv("ESCAPE_SCENE_NAME", "scene_1")
        num_demos = int(os.getenv("ESCAPE_NUM_DEMOS", "1"))

        self.num_envs = num_demos

        cprint(f"\n{'='*60}\n  Loading {num_demos} demo scenes (1 demo = 1 env)\n  Scene: {scene_name}, Start ID: {start_demo_id}\n{'='*60}", "cyan", attrs=['bold'])

        # HDF5 file for initial robot states
        hdf5_file = _get_static_scene_hdf5_path(scene_name)

        # Robot configuration
        if ENABLE_RADAR:
            robot_cfg = self._create_radar_robot_cfg()
            print("  [OK] Using Franka with radar")
        else:
            robot_cfg = FRANKA_PANDA_CFG.copy()
            print("  Using standard Franka")

        robot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"
        robot_cfg.init_state = FRANKA_PANDA_CFG.init_state.replace(pos=(0.0, 0.0, 0.0))

        # # Load initial state for first demo
        # initial_joints, gripper_state = self._load_initial_robot_state(hdf5_file, start_demo_id)
        # if initial_joints is not None:
        #     robot_cfg.init_state.joint_pos = {
        #         "panda_joint1": float(initial_joints[0]),
        #         "panda_joint2": float(initial_joints[1]),
        #         "panda_joint3": float(initial_joints[2]),
        #         "panda_joint4": float(initial_joints[3]),
        #         "panda_joint5": float(initial_joints[4]),
        #         "panda_joint6": float(initial_joints[5]),
        #         "panda_joint7": float(initial_joints[6]),
        #         "panda_finger_joint.*": float(gripper_state),
        #     }

        # High-stiffness PD control
        robot_cfg.actuators["panda_shoulder"] = ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit=400.0,
            velocity_limit=10.0,
            stiffness=10000.0,
            damping=200.0,
            armature=0,
        )
        robot_cfg.actuators["panda_forearm"] = ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit=200.0,
            velocity_limit=10.0,
            stiffness=10000.0,
            damping=200.0,
            armature=0,
        )
        self.robot = robot_cfg

        # End-effector frame transformer
        self.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=FRAME_MARKER_CFG.replace(prim_path="/Visuals/EEFrameTransformer"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="ee_tcp",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.1034)),
                )
            ],
        )

        # Load USD scenes for each environment
        for env_idx in range(num_demos):
            cur_demo_id = start_demo_id + env_idx
            print(f"\n  Loading Env {env_idx} with Demo ID {cur_demo_id}")
            usd_path = _get_usd_scene_path(scene_name, cur_demo_id)

            if not os.path.exists(usd_path):
                cprint(f"  [WARN] USD file not found: {usd_path}", "yellow")
                continue

            # Use a fixed prim path; IsaacLab does not expand {ENV_REGEX_NS} here.
            obstacle_name = f"obstacles_demo_{cur_demo_id}"
            obstacle_cfg = AssetBaseCfg(
                prim_path=f"/World/envs/env_{env_idx}/obstacles",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=usd_path,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True,
                        disable_gravity=True,
                    ) if ENABLE_RADAR else None,
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        collision_enabled=True if ENABLE_RADAR else False,
                    ),
                    # Repository-local Oak Planks material. Keep the sibling texture folder with the MDL.
                    visual_material=sim_utils.MdlFileCfg(
                        mdl_path=_get_obstacle_wood_mdl_path(),
                        project_uvw=True,
                        texture_scale=(0.6, 0.6),
                        albedo_brightness=1.0,
                    ),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(0.0, 0.0, 0.0),
                    rot=(1.0, 0.0, 0.0, 0.0),
                ),
            )

            # Include the demo id in the attribute name to keep it unique.
            setattr(self, obstacle_name, obstacle_cfg)

            cprint(f"  [OK] Env {env_idx} loaded demo {cur_demo_id}: {os.path.basename(usd_path)}", "green")

        cprint(f"\n  Loaded {num_demos} USD scenes\n{'='*60}\n", "cyan")

    def _create_radar_robot_cfg(self) -> ArticulationCfg:
        """Create Franka robot with radar sensor"""
        return ArticulationCfg(
            spawn=sim_utils.UsdFileCfg(
                usd_path="assets/radar_franka/FrankaEmika/panda_instanceable_radar.usd",
                activate_contact_sensors=False,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=5.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    "panda_joint1": 0.0,
                    "panda_joint2": -0.569,
                    "panda_joint3": 0.0,
                    "panda_joint4": -2.810,
                    "panda_joint5": 0.0,
                    "panda_joint6": 3.037,
                    "panda_joint7": 0.741,
                    "panda_finger_joint.*": 0.04,
                },
            ),
            actuators={
                "panda_shoulder": ImplicitActuatorCfg(
                    joint_names_expr=["panda_joint[1-4]"],
                    effort_limit=400.0,
                    velocity_limit=10.0,
                    stiffness=100000.0,
                    damping=5000.0,
                    armature=0,
                ),
                "panda_forearm": ImplicitActuatorCfg(
                    joint_names_expr=["panda_joint[5-6]"],
                    effort_limit=200.0,
                    velocity_limit=10.0,
                    stiffness=80000.0,
                    damping=3000.0,
                    armature=0,
                ),
                "panda_wrist": ImplicitActuatorCfg(
                    joint_names_expr=["panda_joint7"],
                    effort_limit=200.0,
                    velocity_limit=10.0,
                    stiffness=200000.0,
                    damping=8000.0,
                    armature=0,
                ),
                "panda_hand": ImplicitActuatorCfg(
                    joint_names_expr=["panda_finger_joint.*"],
                    effort_limit_sim=200.0,
                    stiffness=2e3,
                    damping=1e2,
                ),
            },
            soft_joint_pos_limit_factor=1.0,
        )

    def _load_initial_robot_state(self, hdf5_file: str, demo_id: int):
        """Load initial robot state from HDF5"""
        with h5py.File(hdf5_file, "r") as f:
            demo = f["data"][f"demo_{demo_id}"]
            initial_joints = demo["obs"]["current_angles"][0]
            gripper_state = 0.04

        print(f"  Loaded initial state from demo_{demo_id}")
        return initial_joints, gripper_state


# MDP Configuration
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()


@configclass
class ActionsCfg:
    arm_action: ActionTermCfg = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale=1.0,
        use_default_offset=True,
    )


@configclass
class EventCfg:
    """Only reset robot state, no scene switching"""
    reset_robot = EventTerm(
        func=hdf5_events.reset_robot_only_usd,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class RewardsCfg:
    pass


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


# Environment Configuration
@configclass
class FrankaHDF5Env_USD_ALL_Cfg(ManagerBasedRLEnvCfg):
    scene: FrankaUSDSceneCfg_ALL = FrankaUSDSceneCfg_ALL(num_envs=1, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.sim.dt = 1.0 / 120.0
        self.decimation = 40
        self.episode_length_s = 10.0
        self.viewer.eye = (-10.0, 0.0, 8)
        self.viewer.lookat = (0.0, 0.0, 0.0)

        # Configure for radar mode if enabled
        if os.getenv("ESCAPE_ENABLE_RADAR", "false").lower() in ("true", "1", "yes"):
            self.sim.use_fabric = False
            self.sim.device = "cpu"
            print("[joint_pos_env_cfg_usd] use_fabric=False, device=cpu (radar mode)")
