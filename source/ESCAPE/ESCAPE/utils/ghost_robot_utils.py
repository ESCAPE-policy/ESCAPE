"""Utilities for visualizing goal robot configuration as semi-transparent ghost."""

import os
import numpy as np
from scipy.spatial.transform import Rotation as R

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from pxr import Sdf, Usd, UsdShade


_ghost_markers: dict[int, VisualizationMarkers] = {}
_ghost_marker_keys: dict[int, tuple] = {}

# Real Franka link mesh USD paths.
_FRANKA_ASSET_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../assets/Franka_3/FrankaEmika/Props"))


def _bind_material_to_ghost_meshes(stage, env_id: int, material: UsdShade.Material) -> int:
    root = stage.GetPrimAtPath(f"/World/Visuals/GhostFranka_{env_id}")
    if not root.IsValid():
        return 0

    bound = 0
    for prim in Usd.PrimRange(root):
        if prim.GetTypeName() not in {"Mesh", "Cube", "Capsule", "Sphere"}:
            continue
        UsdShade.MaterialBindingAPI(prim).Bind(
            material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        )
        bound += 1
    return bound


def _bind_mdl_material_to_ghost(stage, env_id: int, mdl_path: str, mdl_material: str | None = None) -> int:
    mdl_path = os.path.abspath(os.path.expanduser(mdl_path))
    material_name = mdl_material or os.path.splitext(os.path.basename(mdl_path))[0]
    material_path = Sdf.Path(f"/World/Looks/GhostFrankaMDL_{env_id}")
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, material_path.AppendPath("Shader"))
    shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    shader.SetSourceAsset(Sdf.AssetPath(mdl_path), "mdl")
    shader.SetSourceAssetSubIdentifier(material_name, "mdl")
    material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "out")
    return _bind_material_to_ghost_meshes(stage, env_id, material)


def _bind_preview_material_to_ghost(
    stage,
    env_id: int,
    opacity: float,
    color=(1.0, 1.0, 1.0),
    emissive_strength: float = 0.0,
) -> int:
    opacity = float(np.clip(opacity, 0.0, 1.0))
    color = tuple(float(c) for c in color)
    emissive_color = tuple(float(np.clip(c * emissive_strength, 0.0, 1.0)) for c in color)
    material_path = Sdf.Path(f"/World/Looks/GhostFrankaPreview_{env_id}")
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, material_path.AppendPath("PreviewSurface"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(emissive_color)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return _bind_material_to_ghost_meshes(stage, env_id, material)
def compute_franka_fk_all_links(goal_joints, base_transform=None):
    """Compute forward kinematics for ALL Franka links."""
    dh_params = [
        [0, 0.333, 0],
        [0, 0, -np.pi/2],
        [0, 0.316, np.pi/2],
        [0.0825, 0, np.pi/2],
        [-0.0825, 0.384, -np.pi/2],
        [0, 0, np.pi/2],
        [0.088, 0, np.pi/2],
        [0, 0.107, 0],
    ]
    
    T = np.eye(4) if base_transform is None else base_transform.copy()
    transforms = [T.copy()]  # link0
    
    for i in range(7):
        a, d, alpha = dh_params[i]
        theta = goal_joints[i]
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        
        T_joint = np.array([
            [ct, -st, 0, a],
            [st*ca, ct*ca, -sa, -sa*d],
            [st*sa, ct*sa, ca, ca*d],
            [0, 0, 0, 1]
        ])
        T = T @ T_joint
        transforms.append(T.copy())  # link1-7
    
    # Link8 offset (fixed joint from link7 to link8)
    T_link8 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0.107],[0,0,0,1]])
    T = T @ T_link8
    
    # Hand rotation around Z from URDF: rpy="0 0 -0.785398163397".
    angle = -np.pi / 4
    R_z = np.array([
        [np.cos(angle), -np.sin(angle), 0, 0],
        [np.sin(angle),  np.cos(angle), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    T_hand = T @ R_z
    transforms.append(T_hand.copy())  # hand
    
    # Fingers (from hand frame, offset in Z = 0.0584)
    # Left finger: Y = +gripper_width/2 (default ~0.04 for open)
    # Right finger: Y = -gripper_width/2, rotated 180 deg around Z
    finger_z = 0.0584
    
    # Left finger
    T_left = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0.04],  # Y offset
        [0, 0, 1, finger_z],
        [0, 0, 0, 1]
    ])
    transforms.append(T_hand @ T_left)  # left finger
    
    # Right finger: rotated 180 deg around Z (from URDF: rpy="0 0 3.14159265359")
    angle_rf = np.pi
    R_z_180 = np.array([
        [np.cos(angle_rf), -np.sin(angle_rf), 0, 0],
        [np.sin(angle_rf),  np.cos(angle_rf), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    T_right = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, -0.04],  # Y offset (negative)
        [0, 0, 1, finger_z],
        [0, 0, 0, 1]
    ])
    transforms.append(T_hand @ T_right @ R_z_180)  # right finger
    
    return transforms  # 11 transforms total


def _create_ghost_visual_material(
    opacity: float = 0.3,
    mdl_path: str | None = None,
    color=(1.0, 1.0, 1.0),
    emissive_strength: float = 0.0,
):
    if mdl_path:
        return sim_utils.MdlFileCfg(mdl_path=os.path.abspath(os.path.expanduser(mdl_path)))

    opacity = float(np.clip(opacity, 0.0, 1.0))
    color = tuple(float(c) for c in color)
    emissive_color = tuple(float(np.clip(c * emissive_strength, 0.0, 1.0)) for c in color)
    return sim_utils.PreviewSurfaceCfg(
        diffuse_color=color,
        emissive_color=emissive_color,
        opacity=opacity,
    )


def _create_ghost_marker_cfg(
    env_id: int,
    opacity: float = 0.3,
    mdl_path: str | None = None,
    color=(1.0, 1.0, 1.0),
    emissive_strength: float = 0.0,
) -> VisualizationMarkersCfg:
    """Create marker config with REAL Franka link meshes as prototypes."""
    ghost_visual_material = _create_ghost_visual_material(
        opacity=opacity,
        mdl_path=mdl_path,
        color=color,
        emissive_strength=emissive_strength,
    )

    return VisualizationMarkersCfg(
        prim_path=f"/World/Visuals/GhostFranka_{env_id}",
        markers={
            # Use real Franka link mesh files.
            "link0": sim_utils.UsdFileCfg(
                usd_path=f"{_FRANKA_ASSET_BASE}/panda_link0.usd",
                visual_material=ghost_visual_material,
            ),
            "link1": sim_utils.UsdFileCfg(
                usd_path=f"{_FRANKA_ASSET_BASE}/panda_link1.usd",
                visual_material=ghost_visual_material,
            ),
            "link2": sim_utils.UsdFileCfg(
                usd_path=f"{_FRANKA_ASSET_BASE}/panda_link2.usd",
                visual_material=ghost_visual_material,
            ),
            "link3": sim_utils.UsdFileCfg(
                usd_path=f"{_FRANKA_ASSET_BASE}/panda_link3.usd",
                visual_material=ghost_visual_material,
            ),
            "link4": sim_utils.UsdFileCfg(
                usd_path=f"{_FRANKA_ASSET_BASE}/panda_link4.usd",
                visual_material=ghost_visual_material,
            ),
            "link5": sim_utils.UsdFileCfg(
                usd_path=f"{_FRANKA_ASSET_BASE}/panda_link5.usd",
                visual_material=ghost_visual_material,
            ),
            "link6": sim_utils.UsdFileCfg(
                usd_path=f"{_FRANKA_ASSET_BASE}/panda_link6.usd",
                visual_material=ghost_visual_material,
            ),
            "link7": sim_utils.UsdFileCfg(
                usd_path=f"{_FRANKA_ASSET_BASE}/panda_link7.usd",
                visual_material=ghost_visual_material,
            ),
            "hand": sim_utils.UsdFileCfg(
                usd_path=f"{_FRANKA_ASSET_BASE}/panda_hand.usd",
                visual_material=ghost_visual_material,
            ),
            "left_finger": sim_utils.UsdFileCfg(
                usd_path=f"{_FRANKA_ASSET_BASE}/panda_leftfinger.usd",
                visual_material=ghost_visual_material,
            ),
            "right_finger": sim_utils.UsdFileCfg(
                usd_path=f"{_FRANKA_ASSET_BASE}/panda_rightfinger.usd",
                visual_material=ghost_visual_material,
            ),
        },
    )


def show_goal_as_ghost(
    stage,
    env_id: int,
    goal_joints,
    opacity: float = 0.3,
    mdl_path: str | None = None,
    mdl_material: str | None = None,
    color=(1.0, 1.0, 1.0),
    emissive_strength: float = 0.0,
):
    """Display goal configuration using REAL link meshes."""
    from pxr import UsdGeom
    
    global _ghost_markers
    global _ghost_marker_keys
    
    # Get robot base transform
    robot_prim_path = f"/World/envs/env_{env_id}/Robot"
    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    if not robot_prim.IsValid():
        print(f"Error: Robot not found at {robot_prim_path}")
        return
    
    robot_xform = UsdGeom.Xformable(robot_prim)
    m = robot_xform.ComputeLocalToWorldTransform(0)
    base_transform = np.array([[m[i][j] for j in range(4)] for i in range(4)])
    
    # Compute FK for all 11 links
    transforms = compute_franka_fk_all_links(goal_joints, base_transform)
    
    # Extract positions and orientations
    num_links = len(transforms)
    positions = np.zeros((num_links, 3), dtype=np.float32)
    orientations = np.zeros((num_links, 4), dtype=np.float32)
    
    for i, T in enumerate(transforms):
        positions[i] = T[:3, 3]
        quat = R.from_matrix(T[:3, :3]).as_quat()  # [x, y, z, w]
        orientations[i] = [quat[3], quat[0], quat[1], quat[2]]  # [w, x, y, z]
    
    material_key = (
        "mdl" if mdl_path else "preview",
        os.path.abspath(os.path.expanduser(mdl_path)) if mdl_path else None,
        mdl_material,
        round(float(opacity), 4),
        tuple(float(c) for c in color),
        round(float(emissive_strength), 4),
    )

    if env_id not in _ghost_markers or _ghost_marker_keys.get(env_id) != material_key:
        cfg = _create_ghost_marker_cfg(
            env_id,
            opacity=opacity,
            mdl_path=mdl_path,
            color=color,
            emissive_strength=emissive_strength,
        )
        _ghost_markers[env_id] = VisualizationMarkers(cfg)
        _ghost_marker_keys[env_id] = material_key
        print(f"[OK] Created ghost with real Franka meshes for env {env_id} (opacity={opacity:.2f})")

    if mdl_path:
        print(f"[OK] Created ghost prototypes with MDL material: {mdl_path}")
    else:
        _bind_preview_material_to_ghost(
            stage,
            env_id,
            opacity=opacity,
            color=color,
            emissive_strength=emissive_strength,
        )
    
    marker = _ghost_markers[env_id]
    
    # marker_indices selects the matching link prototype for each position.
    marker_indices = np.arange(num_links, dtype=np.int32)
    
    marker.visualize(
        translations=positions,
        orientations=orientations,
        marker_indices=marker_indices,
    )
    
    print(f"[OK] Ghost updated (env {env_id}) - {num_links} real link meshes")
