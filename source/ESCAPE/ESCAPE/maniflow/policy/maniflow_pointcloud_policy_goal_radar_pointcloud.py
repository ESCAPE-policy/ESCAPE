from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from termcolor import cprint
import yaml

from maniflow.mpiformer.geometry import TorchCuboids
from maniflow.model.common.normalizer import LinearNormalizer
from maniflow.policy.base_policy import BasePolicy
from maniflow.common.pytorch_util import dict_apply
from maniflow.common.model_util import print_params
from maniflow.model.vision_3d.pointnet_extractor import DP3Encoder
from maniflow.model.diffusion.ditx import DiTX
from maniflow.model.common.sample_util import *
from robofin.samplers import TorchFrankaSampler
from maniflow.mpiformer.loss import collision_loss


class ManiFlowTransformerPointcloudPolicyForce(BasePolicy):
    def __init__(self, 
            shape_meta: dict,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_timestep_embed_dim=256,
            diffusion_target_t_embed_dim=256,
            visual_cond_len=1024,
            n_layer=3,
            n_head=4,
            n_emb=256,
            qkv_bias=False,
            qk_norm=False,
            block_type="DiTX",
            encoder_type="DP3Encoder",
            encoder_output_dim=256,
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="pointnet",
            pointcloud_encoder_cfg=None,
            downsample_points=False,
            pre_norm_modality=False,
            language_conditioned=False,
            # consistency flow training parameters
            flow_batch_ratio=0.75,
            consistency_batch_ratio=0.25,
            denoise_timesteps=10,
            sample_t_mode_flow="beta", 
            sample_t_mode_consistency="discrete",
            sample_dt_mode_consistency="uniform", 
            sample_target_t_mode="relative",
            # Goal conditioning parameters
            use_goal_cond=False,
            goal_dim=7,
            goal_embed_dim=256,
            use_terminal_goal_reaching_loss=False,
            terminal_goal_reaching_loss_weight=0.1,
            terminal_goal_reaching_window_steps=15,
            # Collision loss parameters
            use_collision_loss=False,
            collision_margin=0.03,
            collision_loss_weight=10,
            collision_loss_scale_surface_points=1.0,
            collision_loss_scale_yaml_spheres=0.1,
            collision_loss_scale_dataset_pointcloud=1.0,
            num_robot_points=1024,
            prismatic_joint=0.04,
            collision_t_threshold=0.90,
            collision_num_steps=10,
            collision_geometry_mode="surface_points",
            collision_spheres_yaml_path=None,
            collision_robot_usd_path=None,
            collision_robot_usd_root_path=None,
            collision_dataset_exclude_body_names=None,
            collision_dataset_body_sampling_weights=None,
            **kwargs):
        super().__init__()

        self.collision_t_threshold = collision_t_threshold
        self.collision_num_steps = collision_num_steps 

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2:
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")
            
        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])

        if use_goal_cond:
            goal_mlp_size = (goal_embed_dim, goal_embed_dim)
        else:
            goal_mlp_size = None

        # create observation encoder
        self.encoder_type = encoder_type
        if encoder_type == "DP3Encoder":
            obs_encoder = DP3Encoder(
                observation_space=obs_dict,
                img_crop_shape=crop_shape,
                out_channel=encoder_output_dim,
                pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                use_pc_color=use_pc_color,
                pointnet_type=pointnet_type,
                downsample_points=downsample_points,
                use_goal_cond=use_goal_cond,
                goal_dim=goal_dim,
                goal_mlp_size=goal_mlp_size
            )
        else:
            raise ValueError(f"Unsupported encoder type {encoder_type}")
        
        obs_feature_dim = obs_encoder.output_shape()
        cprint(f"[Policy] obs_feature_dim: {obs_feature_dim}", "yellow")

        self.use_goal_cond = use_goal_cond
        self.goal_dim = goal_dim
        self.goal_embed_dim = goal_embed_dim
        self.use_terminal_goal_reaching_loss = use_terminal_goal_reaching_loss
        self.terminal_goal_reaching_loss_weight = terminal_goal_reaching_loss_weight
        self.terminal_goal_reaching_window_steps = terminal_goal_reaching_window_steps
        cprint(f"[Encoder_type] {encoder_type}", "yellow")
       
        # create ManiFlow model
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[Policy] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[Policy] pointnet_type: {self.pointnet_type}", "yellow")
        
        self.use_collision_loss = use_collision_loss
        self.collision_margin = collision_margin
        self.collision_loss_weight = collision_loss_weight
        self.collision_loss_schedule_multiplier = 1.0
        self.collision_loss_scale_surface_points = collision_loss_scale_surface_points
        self.collision_loss_scale_yaml_spheres = collision_loss_scale_yaml_spheres
        self.collision_loss_scale_dataset_pointcloud = collision_loss_scale_dataset_pointcloud
        self.num_robot_points = num_robot_points
        self.prismatic_joint = prismatic_joint
        self.fk_sampler = None
        self.robot_urdf = None
        self.yaml_collision_spheres = None
        self.dataset_collision_points_by_body = None
        self.collision_geometry_mode = collision_geometry_mode
        if self.collision_geometry_mode not in {"surface_points", "yaml_spheres", "dataset_pointcloud"}:
            raise ValueError(
                "collision_geometry_mode must be one of "
                "{'surface_points', 'yaml_spheres', 'dataset_pointcloud'}, "
                f"got {self.collision_geometry_mode!r}"
            )
        self.collision_spheres_yaml_path = self._resolve_collision_spheres_yaml_path(
            collision_spheres_yaml_path
        )
        self.collision_robot_usd_path = self._resolve_collision_robot_usd_path(
            collision_robot_usd_path
        )
        self.collision_robot_usd_root_path = collision_robot_usd_root_path
        self.collision_dataset_exclude_body_names = list(collision_dataset_exclude_body_names or [])
        default_dataset_body_weights = {
            "panda_link0": 0.2,
            "panda_link1": 0.2,
            "panda_hand": 4.0,
            "panda_leftfinger": 6.0,
            "panda_rightfinger": 6.0,
        }
        if collision_dataset_body_sampling_weights is None:
            self.collision_dataset_body_sampling_weights = default_dataset_body_weights
        else:
            self.collision_dataset_body_sampling_weights = {
                str(k): float(v) for k, v in collision_dataset_body_sampling_weights.items()
            }
        
        if use_collision_loss:
            cprint(f"[Policy] Collision loss enabled:", "green")
            cprint(f"  - margin: {collision_margin}", "green")
            cprint(f"  - weight: {collision_loss_weight}", "green")
            cprint(
                f"  - scale(surface_points): {self.collision_loss_scale_surface_points}",
                "green",
            )
            cprint(
                f"  - scale(yaml_spheres): {self.collision_loss_scale_yaml_spheres}",
                "green",
            )
            cprint(
                f"  - scale(dataset_pointcloud): {self.collision_loss_scale_dataset_pointcloud}",
                "green",
            )
            cprint(f"  - robot points: {num_robot_points}", "green")
            cprint(f"  - num_steps: {collision_num_steps}", "green")
            cprint(f"  - geometry_mode: {self.collision_geometry_mode}", "green")
            if self.collision_geometry_mode == "yaml_spheres":
                cprint(
                    f"  - spheres_yaml: {self.collision_spheres_yaml_path}",
                    "green",
                )
            elif self.collision_geometry_mode == "dataset_pointcloud":
                cprint(
                    f"  - robot_usd: {self.collision_robot_usd_path}",
                    "green",
                )
                cprint(
                    f"  - usd_root: {self.collision_robot_usd_root_path}",
                    "green",
                )
        if self.use_terminal_goal_reaching_loss:
            cprint("[Policy] Terminal goal reaching loss enabled:", "green")
            cprint(
                f"  - weight: {self.terminal_goal_reaching_loss_weight}",
                "green",
            )
            cprint(
                f"  - window_steps: {self.terminal_goal_reaching_window_steps}",
                "green",
            )

        cprint(f"[Policy] Using DiTX model", "red")
        model = DiTX(
            input_dim=input_dim,
            output_dim=action_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            cond_dim=global_cond_dim,
            visual_cond_len=visual_cond_len,
            diffusion_timestep_embed_dim=diffusion_timestep_embed_dim,
            diffusion_target_t_embed_dim=diffusion_target_t_embed_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            block_type=block_type,
            pre_norm_modality=pre_norm_modality,
            language_conditioned=language_conditioned,
        )
        
        self.obs_encoder = obs_encoder
        self.model = model
        
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.language_conditioned = language_conditioned
        self.kwargs = kwargs

        self.num_inference_steps = num_inference_steps
        self.flow_batch_ratio = flow_batch_ratio
        self.consistency_batch_ratio = consistency_batch_ratio
        assert flow_batch_ratio + consistency_batch_ratio == 1.0
        self.denoise_timesteps = denoise_timesteps
        self.sample_t_mode_flow = sample_t_mode_flow
        self.sample_t_mode_consistency = sample_t_mode_consistency
        self.sample_dt_mode_consistency = sample_dt_mode_consistency
        self.sample_target_t_mode = sample_target_t_mode
        assert self.sample_target_t_mode in ["absolute", "relative"]

        cprint(f"[Policy] Initialized with parameters:", "yellow")
        cprint(f"  - horizon: {self.horizon}", "yellow")
        cprint(f"  - n_action_steps: {self.n_action_steps}", "yellow")
        cprint(f"  - n_obs_steps: {self.n_obs_steps}", "yellow")
        cprint(f"  - num_inference_steps: {self.num_inference_steps}", "yellow")
        if self.use_goal_cond:
            cprint(f"  - Goal conditioning: {self.goal_dim}D", "yellow")

        print_params(self)

    def set_collision_loss_schedule_multiplier(self, multiplier: float) -> None:
        self.collision_loss_schedule_multiplier = float(multiplier)
    
    def _init_fk_sampler(self):
        """Lazily initialize the FK sampler."""
        if self.fk_sampler is not None:
            return
        if self.device is None:
            device = "cpu"
            cprint("[Policy] Warning: device is None, using cpu for FK sampler", "yellow")
        else:
            device = self.device
            cprint(f"[Policy] Using device: {device}", "green")

        cprint("[Policy] FK sampler starting initialization", "green")
        self.fk_sampler = TorchFrankaSampler(
            num_robot_points=self.num_robot_points,
            num_eef_points=128,
            device=device,
            with_base_link=False,
            use_cache=True,
        )
        cprint("[Policy] FK sampler initialized for Franka Panda", "green")

    def _resolve_collision_spheres_yaml_path(self, yaml_path: Optional[str]) -> Optional[str]:
        repo_root = Path(__file__).resolve().parents[5]
        if yaml_path:
            candidate = Path(yaml_path).expanduser()
            if not candidate.is_absolute():
                candidate = repo_root / candidate
            if candidate.is_file():
                return str(candidate)
            remapped = self._remap_legacy_asset_path(candidate, repo_root)
            if remapped is not None:
                return str(remapped)
            return str(candidate)

        candidate_paths = [
            repo_root / "assets" / "franka_radar_mesh_for_point_cloud_filter.yml",
            repo_root / "assets" / "franka_radar_mesh_detailed.yml",
        ]
        for candidate in candidate_paths:
            if candidate.is_file():
                return str(candidate)
        return None

    def _resolve_collision_robot_usd_path(self, usd_path: Optional[str]) -> Optional[str]:
        repo_root = Path(__file__).resolve().parents[5]
        if usd_path:
            candidate = Path(usd_path).expanduser()
            if not candidate.is_absolute():
                candidate = repo_root / candidate
            if candidate.is_file():
                return str(candidate)
            remapped = self._remap_legacy_asset_path(candidate, repo_root)
            if remapped is not None:
                return str(remapped)
            return str(candidate)

        candidate_paths = [
            repo_root / "assets" / "radar_franka" / "FrankaEmika" / "panda_instanceable_radar.usd",
            repo_root / "assets" / "radar_franka" / "FrankaEmika" / "panda_instanceable_radar_v6.usd",
            repo_root / "assets" / "radar_franka" / "FrankaEmika" / "panda_instanceable.usd",
        ]
        for candidate in candidate_paths:
            if candidate.is_file():
                return str(candidate)
        return None

    @staticmethod
    def _remap_legacy_asset_path(candidate: Path, repo_root: Path) -> Optional[Path]:
        parts = candidate.parts
        if "assert" not in parts:
            return None
        idx = parts.index("assert")
        remapped = repo_root / "assets" / Path(*parts[idx + 1:])
        return remapped if remapped.is_file() else None

    def _load_collision_spheres_from_yaml(self, yaml_path: str) -> Dict[str, Dict[str, torch.Tensor]]:
        with open(yaml_path, "r", encoding="utf-8") as f:
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

            if not centers:
                continue

            parsed[link_name] = {
                "centers": torch.tensor(centers, dtype=torch.float32, device=self.device),
                "radii": torch.tensor(radii, dtype=torch.float32, device=self.device),
            }

        if not parsed:
            raise ValueError(f"No valid collision spheres parsed from {yaml_path}")
        return parsed

    def _init_robot_urdf(self):
        if self.robot_urdf is not None:
            return

        from robofin.robot_constants import FrankaConstants
        from robofin.torch_urdf import TorchURDF

        self.robot_urdf = TorchURDF.load(
            FrankaConstants.urdf,
            lazy_load_meshes=True,
            device=self.device,
        )

    def _init_yaml_collision_model(self):
        if self.yaml_collision_spheres is not None:
            return
        if not self.collision_spheres_yaml_path:
            raise FileNotFoundError(
                "collision_geometry_mode='yaml_spheres' but no collision_spheres_yaml_path "
                "was provided and no default YAML file was found."
            )

        yaml_path = Path(self.collision_spheres_yaml_path).expanduser()
        if not yaml_path.is_file():
            raise FileNotFoundError(f"Collision spheres YAML not found: {yaml_path}")

        # Import robofin lazily so training paths that do not use collision
        # geometry do not trigger ESCAPE package side effects.
        self._init_robot_urdf()
        self.yaml_collision_spheres = self._load_collision_spheres_from_yaml(str(yaml_path))
        sphere_count = sum(
            sphere_info["centers"].shape[0]
            for sphere_info in self.yaml_collision_spheres.values()
        )
        cprint(
            f"[Policy] YAML collision spheres initialized: {sphere_count} spheres "
            f"from {yaml_path}",
            "green",
        )

    def _find_robot_usd_root_prim(self, stage, body_names: List[str]):
        if self.collision_robot_usd_root_path:
            candidate = stage.GetPrimAtPath(self.collision_robot_usd_root_path)
            if candidate.IsValid():
                return candidate
            raise ValueError(
                "collision_robot_usd_root_path was provided but is invalid: "
                f"{self.collision_robot_usd_root_path}"
            )

        best_prim = None
        best_match_count = -1
        for prim in stage.Traverse():
            if not prim.IsValid():
                continue
            child_names = {child.GetName() for child in prim.GetChildren()}
            match_count = sum(1 for body_name in body_names if body_name in child_names)
            if match_count > best_match_count:
                best_prim = prim
                best_match_count = match_count

        if best_prim is None or best_match_count <= 0:
            raise ValueError(
                "Failed to infer robot root prim from USD. Please provide "
                "collision_robot_usd_root_path explicitly."
            )
        return best_prim

    def _sample_dataset_collision_points_from_usd(self) -> Dict[str, torch.Tensor]:
        if not self.collision_robot_usd_path:
            raise FileNotFoundError(
                "collision_geometry_mode='dataset_pointcloud' but no collision_robot_usd_path "
                "was provided and no default robot USD was found."
            )

        usd_path = Path(self.collision_robot_usd_path).expanduser()
        if not usd_path.is_file():
            raise FileNotFoundError(f"Collision robot USD not found: {usd_path}")

        try:
            import trimesh
            from pxr import Usd
            from maniflow.common.usd_sampling_utils import (
                compute_relative_transform_matrix,
                iter_descendant_geometry_prims,
                prim_to_trimesh_local,
            )
        except ImportError as exc:
            raise ImportError(
                "dataset_pointcloud collision geometry requires 'pxr', trimesh, and "
                "maniflow.common.usd_sampling_utils to be available in the runtime environment."
            ) from exc

        stage = Usd.Stage.Open(str(usd_path))
        if stage is None:
            raise RuntimeError(f"Failed to open robot USD stage: {usd_path}")

        body_names = [
            link.name for link in self.robot_urdf.links
            if link.name.startswith("panda_")
        ]
        robot_root_prim = self._find_robot_usd_root_prim(stage, body_names)

        geometry_entries = []
        excluded_bodies = set(self.collision_dataset_exclude_body_names)
        body_sampling_weights = self.collision_dataset_body_sampling_weights

        for body_name in body_names:
            if body_name in excluded_bodies:
                continue

            link_prim = stage.GetPrimAtPath(f"{robot_root_prim.GetPath()}/{body_name}")
            if not link_prim.IsValid():
                continue

            sampling_weight = max(float(body_sampling_weights.get(body_name, 1.0)), 0.0)
            if sampling_weight <= 0.0:
                continue

            for geom_prim in iter_descendant_geometry_prims(link_prim):
                try:
                    mesh_local = prim_to_trimesh_local(geom_prim)
                    transform_link_to_geom = compute_relative_transform_matrix(link_prim, geom_prim)
                    mesh_in_link = mesh_local.copy()
                    mesh_in_link.apply_transform(transform_link_to_geom)
                    area = float(mesh_in_link.area)
                except Exception:
                    continue

                if area <= 0.0:
                    continue

                geometry_entries.append(
                    {
                        "body_name": body_name,
                        "mesh_local": mesh_local,
                        "transform_link_to_geom": transform_link_to_geom,
                        "weighted_area": area * sampling_weight,
                    }
                )

        if not geometry_entries:
            raise ValueError(
                f"No valid robot geometries were sampled from USD: {usd_path}"
            )

        areas = torch.tensor(
            [entry["weighted_area"] for entry in geometry_entries],
            dtype=torch.float64,
        ).cpu().numpy()
        total_area = max(float(areas.sum()), 1e-8)
        points_per_geom = (areas / total_area * int(self.num_robot_points)).astype(int)
        remaining = int(self.num_robot_points) - int(points_per_geom.sum())
        if remaining > 0:
            largest_indices = areas.argsort()[::-1][:remaining]
            points_per_geom[largest_indices] += 1

        points_by_body = {}
        geometry_count = 0
        for entry, geom_points in zip(geometry_entries, points_per_geom):
            if geom_points <= 0:
                continue

            sampled_local = entry["mesh_local"].sample(int(geom_points)).astype("float32")
            sampled_local_h = torch.cat(
                [
                    torch.as_tensor(sampled_local, dtype=torch.float32),
                    torch.ones((sampled_local.shape[0], 1), dtype=torch.float32),
                ],
                dim=1,
            )
            transform_link_to_geom = torch.as_tensor(
                entry["transform_link_to_geom"], dtype=torch.float32
            )
            sampled_in_link = (transform_link_to_geom @ sampled_local_h.T).T[:, :3]

            body_name = entry["body_name"]
            points_by_body.setdefault(body_name, []).append(sampled_in_link)
            geometry_count += 1

        dataset_points_by_body = {
            body_name: torch.cat(points_list, dim=0).to(self.device)
            for body_name, points_list in points_by_body.items()
            if points_list
        }
        total_points = sum(points.shape[0] for points in dataset_points_by_body.values())
        cprint(
            "[Policy] Dataset robot pointcloud initialized: "
            f"{total_points} pts, {len(dataset_points_by_body)} links, {geometry_count} geometries "
            f"from {usd_path}",
            "green",
        )
        cprint(
            f"[Policy] Dataset pointcloud root prim: {robot_root_prim.GetPath()}",
            "green",
        )
        return dataset_points_by_body

    def _init_dataset_collision_model(self):
        if self.dataset_collision_points_by_body is not None:
            return

        self._init_robot_urdf()
        self.dataset_collision_points_by_body = self._sample_dataset_collision_points_from_usd()

    def _get_collision_geometry_scale(self) -> float:
        if self.collision_geometry_mode == "yaml_spheres":
            return self.collision_loss_scale_yaml_spheres
        if self.collision_geometry_mode == "dataset_pointcloud":
            return self.collision_loss_scale_dataset_pointcloud
        return self.collision_loss_scale_surface_points

    def _prepare_key_actions_fk(self, actions_pred: torch.Tensor) -> torch.Tensor:
        key_actions = actions_pred[..., :7].reshape(-1, 7)
        expected_dof = len(self.robot_urdf.actuated_joints)
        current_dof = key_actions.shape[1]
        if current_dof < expected_dof:
            pad = self.prismatic_joint * torch.ones(
                (key_actions.shape[0], expected_dof - current_dof),
                dtype=key_actions.dtype,
                device=key_actions.device,
            )
            return torch.cat([key_actions, pad], dim=1)
        if current_dof > expected_dof:
            return key_actions[:, :expected_dof]
        return key_actions

    def _encode_obs(self, nobs: Dict, goal: torch.Tensor = None) -> torch.Tensor:
        """
        Encode normalized point-cloud observations.

        Args:
            nobs: normalized observations dict with 'point_cloud' and 'agent_pos'
            goal: normalized goal tensor [B*T, goal_dim]

        Returns:
            vis_cond: [B, n_obs_steps*L, obs_feature_dim]
        """
        batch_size = nobs['point_cloud'].shape[0]
        
        # Reshape for encoder: [B, T, ...] -> [B*T, ...]
        this_nobs = dict_apply(nobs, 
            lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]).to(self.device))
        
        # Encode point cloud + agent_pos (+ goal if enabled)
        if self.use_goal_cond and goal is not None:
            goal_expanded = goal.unsqueeze(1).expand(-1, self.n_obs_steps, -1).reshape(-1, self.goal_dim)
            obs_features = self.obs_encoder(this_nobs, goal=goal_expanded)
        else:
            obs_features = self.obs_encoder(this_nobs)
        
        # obs_features: [B*n_obs_steps, L, D_obs]

        # Reshape: [B*n_obs_steps, L, D] -> [B, n_obs_steps*L, D]
        D = obs_features.shape[-1]
        vis_cond = obs_features.reshape(batch_size, -1, D)
        
        return vis_cond

    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, 
            vis_cond=None,
            lang_cond=None,
            **kwargs
            ):
        
        noise = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=None)
        
        ode_traj = self.sample_ode(
            x0 = noise, 
            N = self.num_inference_steps,
            vis_cond=vis_cond,
            lang_cond=lang_cond,
           **kwargs)
        
        return ode_traj[-1]

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "point_cloud" and "agent_pos"; "goal" is required when goal conditioning is enabled
        """
        # Normalize input
        nobs = self.normalizer.normalize(obs_dict)
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        
        # Extract and normalize goal
        goal = None
        if self.use_goal_cond:
            if 'goal' not in obs_dict:
                raise ValueError("Goal required but not in obs_dict")
            goal = self.normalizer['goal'].normalize(obs_dict['goal']).to(self.device, dtype=self.dtype)
        
        B, To = next(iter(nobs.values())).shape[:2]
        T = self.horizon
        Da = self.action_dim
        device = self.device
        dtype = self.dtype
    
        # Encode observations.
        vis_cond = self._encode_obs(nobs, goal)
        
        # Language condition
        lang_cond = None
        if self.language_conditioned:
            lang_cond = nobs.get('task_name', None)
    
        # Empty data for action
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)

        # Run sampling
        nsample = self.conditional_sample(
            cond_data, 
            vis_cond=vis_cond,
            lang_cond=lang_cond,
            **self.kwargs)
        
        # Unnormalize prediction
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # Get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred,
        }
        
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
    
    def get_optimizer(
            self, 
            lr: float,
            weight_decay: float,
            obs_encoder_lr: float = None,
            obs_encoder_weight_decay: float = None,
            betas: Tuple[float, float] = (0.9, 0.95)
        ) -> torch.optim.Optimizer:
        optim_groups = self.model.get_optim_groups(weight_decay=weight_decay)
        
        backbone_params = list()
        other_obs_params = list()
        if obs_encoder_lr is not None:
            cprint(f"[Policy] Use different lr for obs_encoder: {obs_encoder_lr}", "yellow")
            for key, value in self.obs_encoder.named_parameters():
                if key.startswith('key_model_map'):
                    backbone_params.append(value)
                else:
                    other_obs_params.append(value)
            optim_groups.append({
                "params": backbone_params,
                "weight_decay": obs_encoder_weight_decay,
                "lr": obs_encoder_lr
            })
            optim_groups.append({
                "params": other_obs_params,
                "weight_decay": obs_encoder_weight_decay
            })
        
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas)
        return optimizer
    
    def sample_t(self, batch_size, mode="uniform"):
        """
        Sample t for flow matching or consistency training.
        """
        if mode == "uniform":
            t = torch.rand((batch_size,), device=self.device)
        elif mode == "lognorm":
            t = sample_logit_normal(batch_size, m=self.lognorm_m, s=self.lognorm_s, device=self.device)
        elif mode == "mode":
            t = sample_mode(batch_size, s=self.mode_s, device=self.device)
        elif mode == "cosmap":
            t = sample_cosmap(batch_size, device=self.device)
        elif mode == "beta":
            t = sample_beta(batch_size, device=self.device)
        elif mode == "discrete":
            t = torch.randint(low=0, high=self.denoise_timesteps, size=(batch_size,)).float()
            t = t / self.denoise_timesteps
        else:
            raise ValueError(f"Unsupported sample_t_mode {mode}")
        return t

    def sample_dt(self, batch_size, sample_dt_mode="uniform"):
        """
        Sample dt for consistency training.
        """
        if sample_dt_mode == "uniform":
            dt = torch.rand((batch_size,), device=self.device)
        else:
            raise ValueError(f"Unsupported sample_dt_mode {sample_dt_mode}")
        return dt
    
    def linear_interpolate(self, noise, target, timestep, epsilon=0.0):
        noise_coeff = 1.0 - (1.0 - epsilon) * timestep
        
        # Linear combination: preserved_noise + scaled_target
        interpolated_data_point = noise_coeff * noise + timestep * target
        return interpolated_data_point

    def get_flow_velocity(self, actions, **model_kwargs):
        target_dict = {}
        
        # get visual and language conditions
        vis_cond = model_kwargs.get('vis_cond', None)
        lang_cond = model_kwargs.get('lang_cond', None)
        flow_batchsize = actions.shape[0]
        device = actions.device
        
        # sample t and dt for flow
        # dt is zero for flow, as we aim to predict the instantaneous velocity at t
        t_flow = self.sample_t(flow_batchsize, mode=self.sample_t_mode_flow).to(device)
        t_flow = t_flow.view(-1, 1, 1)
        dt_flow = torch.zeros((flow_batchsize,), device=device)
        
        # get target timestep
        # target_t_flow is the target timestep for the flow step
        # it can be either absolute or relative to t_flow
        # if absolute, it is t_flow + dt_flow
        # if relative, it is just dt_flow
        if self.sample_target_t_mode == "absolute":
            target_t_flow = t_flow.squeeze() + dt_flow
        elif self.sample_target_t_mode == "relative":
            target_t_flow = dt_flow
        
        # compute interpolated data points at t and predict flow velocity
        x_0_flow = torch.randn_like(actions, device=device) 
        x_1_flow = actions.to(device) 
        x_t_flow = self.linear_interpolate(x_0_flow, x_1_flow, t_flow, epsilon=0.0)
        v_t_flow = x_1_flow - x_0_flow

        target_dict['x_t'] = x_t_flow
        target_dict['t'] = t_flow
        target_dict['target_t'] = target_t_flow
        target_dict['v_target'] = v_t_flow
        target_dict['vis_cond'] = vis_cond
        target_dict['lang_cond'] = lang_cond

        return target_dict
    
    def get_consistency_velocity(self, actions, **model_kwargs):
        """
        Get consistency velocity targets for training.
        Consistency training is used to train the model to be consistent across different timesteps.
        """
        target_dict = {}
        
        # get visual and language conditions
        vis_cond = model_kwargs.get('vis_cond', None)
        lang_cond = model_kwargs.get('lang_cond', None)
        ema_model = model_kwargs.get('ema_model', None)
        consistency_batchsize = actions.shape[0]
        device = actions.device

        # sample t and dt for consistency training
        t_ct = self.sample_t(consistency_batchsize, mode=self.sample_t_mode_consistency).to(device)
        t_ct = t_ct.view(-1, 1, 1)
        delta_t1 = self.sample_dt(consistency_batchsize, sample_dt_mode=self.sample_dt_mode_consistency).to(device)
        # delta_t2 = self.sample_dt(consistency_batchsize, sample_dt_mode=self.sample_dt_mode_consistency).to(device)
        delta_t2 = delta_t1.clone() # use the same delta_t or resample a new one

        # compute next timestep
        t_next = t_ct.squeeze() + delta_t1
        t_next = torch.clamp(t_next, max=1.0) # clip t to ensure it does not exceed 1.0
        t_next = t_next.view(-1, 1, 1)
        
        # compute target timestep
        # target_t_next is the target timestep for the next step
        # it can be either absolute or relative to t_next
        # if absolute, it is t_next + delta_t2
        # if relative, it is just delta_t2
        if self.sample_target_t_mode == "absolute":
            target_t_next = t_next.squeeze() + delta_t2
        elif self.sample_target_t_mode == "relative":
            target_t_next = delta_t2

        # compute interpolated data points at timestep t and t_next
        x0_ct = torch.randn_like(actions, device=device) 
        x1_ct = actions.to(device) 
        x_t_ct = self.linear_interpolate(x0_ct, x1_ct, t_ct, epsilon=0.0)
        x_t_next = self.linear_interpolate(x0_ct, x1_ct, t_next, epsilon=0.0)

        # predict the average velocity from t_next toward next target (t_next + delta_t2)
        with torch.no_grad():
            v_avg_to_next_target = ema_model.model(
                sample=x_t_next, 
                timestep=t_next.squeeze(),
                target_t=target_t_next.squeeze(), 
                vis_cond=vis_cond[-consistency_batchsize:],
                lang_cond=lang_cond[-consistency_batchsize:] if lang_cond is not None else None,
            ) 
        # predict the target data point using the average velocity
        pred_x1_ct = x_t_next + (1 - t_next) * v_avg_to_next_target
        # estimate the velocity at t by using the predicted endpoint
        v_ct = (pred_x1_ct - x_t_ct) / (1 - t_ct)

        # target_t_ct is the target timestep for the current timestep t
        target_t_ct = delta_t1 if self.sample_target_t_mode == "relative" else t_next.squeeze()
        
        target_dict['x_t'] = x_t_ct
        target_dict['t'] = t_ct
        target_dict['target_t'] = target_t_ct
        target_dict['v_target'] = v_ct

        return target_dict
    
    @torch.no_grad()
    def sample_ode(self, x0=None, N=None, **model_kwargs):
        ### NOTE: Use Euler method to sample from the learned flow
        if N is None:
            N = self.num_inference_steps
        dt = 1./N
        traj = [] # to store the trajectory
        x = x0.detach().clone()
        batchsize = x.shape[0]

        t = torch.arange(0, N, device=x0.device, dtype=x0.dtype) / N
        traj.append(x.detach().clone())

        for i in range(N):
            ti = torch.ones((batchsize,), device=self.device) * t[i]
            if self.sample_target_t_mode == "absolute":
                target_t = ti + dt
            elif self.sample_target_t_mode == "relative":
                target_t = dt
            pred = self.model(x, ti, target_t=target_t, **model_kwargs)
            x = x.detach().clone() + pred * dt
            traj.append(x.detach().clone())

        return traj
    
    def _compute_collision_loss_unified(self,
                                         x_t: torch.Tensor,
                                         t: torch.Tensor,
                                         vis_cond: torch.Tensor,
                                         lang_cond: torch.Tensor,
                                         cuboid_centers: torch.Tensor,
                                         cuboid_dims: torch.Tensor,
                                         cuboid_quaternions: torch.Tensor,
                                         num_steps: int = 3,
                                         actions_pred: Optional[torch.Tensor] = None,
                                         ) -> torch.Tensor:
        if self.collision_geometry_mode == "surface_points":
            self._init_fk_sampler()
        elif self.collision_geometry_mode == "yaml_spheres":
            self._init_yaml_collision_model()
        else:
            self._init_dataset_collision_model()
        
        B, horizon, action_dim = x_t.shape
        device = x_t.device
        if actions_pred is None:
            actions_pred = self._rollout_flow_state_to_actions(
                x_t=x_t,
                t=t,
                vis_cond=vis_cond,
                lang_cond=lang_cond,
                num_steps=num_steps,
            )
        
        # Filter valid obstacles
        dim_threshold = 1e-4
        valid_mask = cuboid_dims.abs().sum(dim=-1) > dim_threshold
        num_valid = valid_mask.sum(dim=-1)
        
        if num_valid.sum() == 0:
            return torch.tensor(0.0, device=device)
        
        has_obstacles = num_valid > 0
        if not has_obstacles.all():
            valid_batch_idx = has_obstacles.nonzero(as_tuple=True)[0]
            cuboid_centers = cuboid_centers[valid_batch_idx]
            cuboid_dims = cuboid_dims[valid_batch_idx]
            cuboid_quaternions = cuboid_quaternions[valid_batch_idx]
            actions_pred = actions_pred[valid_batch_idx]
            B = len(valid_batch_idx)

        # Expand obstacles
        cuboid_centers_exp = cuboid_centers.unsqueeze(1).expand(-1, horizon, -1, -1).reshape(B * horizon, -1, 3)
        cuboid_dims_exp = cuboid_dims.unsqueeze(1).expand(-1, horizon, -1, -1).reshape(B * horizon, -1, 3)
        cuboid_quats_exp = cuboid_quaternions.unsqueeze(1).expand(-1, horizon, -1, -1).reshape(B * horizon, -1, 4)

        cuboids_exp = TorchCuboids(
            centers=cuboid_centers_exp,
            dims=cuboid_dims_exp,
            quaternions=cuboid_quats_exp,
        )

        if self.collision_geometry_mode == "yaml_spheres":
            key_actions_fk = self._prepare_key_actions_fk(actions_pred)

            fk = self.robot_urdf.visual_geometry_fk_batch(key_actions_fk, use_names=True)
            sphere_centers_world = []
            sphere_radii = []
            for link_name, sphere_info in self.yaml_collision_spheres.items():
                if link_name not in fk:
                    continue
                transform = fk[link_name]
                centers_local = sphere_info["centers"].unsqueeze(0).expand(key_actions_fk.shape[0], -1, -1)
                radii_local = sphere_info["radii"].unsqueeze(0).expand(key_actions_fk.shape[0], -1)

                rotation = transform[:, :3, :3]
                translation = transform[:, :3, 3]
                centers_world = torch.matmul(centers_local, rotation.transpose(1, 2)) + translation[:, None, :]
                sphere_centers_world.append(centers_world)
                sphere_radii.append(radii_local)

            if not sphere_centers_world:
                return torch.tensor(0.0, device=device)

            sphere_centers_world = torch.cat(sphere_centers_world, dim=1)
            sphere_radii = torch.cat(sphere_radii, dim=1)
            sdf_values_flat = cuboids_exp.sdf(sphere_centers_world) - sphere_radii
            sdf_values = sdf_values_flat.reshape(B, horizon, -1)
        elif self.collision_geometry_mode == "dataset_pointcloud":
            key_actions_fk = self._prepare_key_actions_fk(actions_pred)
            fk = self.robot_urdf.link_fk_batch(key_actions_fk, use_names=True)

            robot_points_world = []
            for body_name, points_local in self.dataset_collision_points_by_body.items():
                if body_name not in fk:
                    continue
                transform = fk[body_name]
                points_local_exp = points_local.unsqueeze(0).expand(key_actions_fk.shape[0], -1, -1)
                rotation = transform[:, :3, :3]
                translation = transform[:, :3, 3]
                points_world = (
                    torch.matmul(points_local_exp, rotation.transpose(1, 2))
                    + translation[:, None, :]
                )
                robot_points_world.append(points_world)

            if not robot_points_world:
                return torch.tensor(0.0, device=device)

            robot_pc_flat = torch.cat(robot_points_world, dim=1)
            sdf_values_flat = cuboids_exp.sdf(robot_pc_flat)
            sdf_values = sdf_values_flat.reshape(B, horizon, -1)
        else:
            # Batched FK surface points.
            key_actions = actions_pred[..., :7].reshape(B * horizon, 7)
            robot_pc_flat = self.fk_sampler.sample(key_actions, self.prismatic_joint)[..., :3]
            N = robot_pc_flat.shape[1]
            sdf_values_flat = cuboids_exp.sdf(robot_pc_flat)
            sdf_values = sdf_values_flat.reshape(B, horizon, N)

        # Collision loss
        penetration = F.relu(self.collision_margin - sdf_values)
        normalized_penetration = 3 * penetration / self.collision_margin
        frame_losses = (normalized_penetration ** 2).mean(dim=-1)
        
        return frame_losses.mean()

    def _rollout_flow_state_to_actions(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        vis_cond: torch.Tensor,
        lang_cond: torch.Tensor,
        num_steps: int,
    ) -> torch.Tensor:
        B = x_t.shape[0]
        t_flat = t.view(B)
        remaining_t = 1.0 - t_flat
        dt_per_sample = remaining_t / num_steps

        x = x_t.clone()
        for i in range(num_steps):
            t_i = t_flat + i * dt_per_sample

            if self.sample_target_t_mode == "absolute":
                target_t = t_i + dt_per_sample
            else:
                target_t = dt_per_sample

            v = self.model(
                sample=x,
                timestep=t_i,
                target_t=target_t,
                vis_cond=vis_cond,
                lang_cond=lang_cond,
            )
            x = x + v * dt_per_sample.view(B, 1, 1)

        return self.normalizer['action'].unnormalize(x)

    def _compute_terminal_goal_reaching_loss(
        self,
        actions_pred: torch.Tensor,
        goal: torch.Tensor,
        remaining_episode_steps: torch.Tensor,
    ) -> torch.Tensor:
        if goal is None:
            raise ValueError(
                "Terminal goal reaching loss requires `goal` in batch."
            )

        window_steps = max(int(self.terminal_goal_reaching_window_steps), 1)
        horizon = actions_pred.shape[1]
        window_steps = min(window_steps, horizon)
        goal_dim = goal.shape[-1]

        remaining_episode_steps = remaining_episode_steps.to(actions_pred.device).view(-1).long()
        active_mask = (remaining_episode_steps >= 1) & (remaining_episode_steps <= window_steps)
        if not active_mask.any():
            return torch.tensor(0.0, device=actions_pred.device)

        actions_goal = actions_pred[active_mask, :, :goal_dim]
        goal_active = goal[active_mask].to(actions_pred.device)
        remaining_active = remaining_episode_steps[active_mask]

        tail_steps = (window_steps - remaining_active + 1).clamp(min=1, max=window_steps)
        step_losses = F.l1_loss(
            actions_goal,
            goal_active.unsqueeze(1).expand(-1, horizon, -1),
            reduction='none',
        ).mean(dim=-1)

        tail_indices = torch.arange(horizon, device=actions_pred.device).unsqueeze(0)
        tail_mask = tail_indices >= (horizon - tail_steps.unsqueeze(1))
        per_sample_loss = (step_losses * tail_mask.float()).sum(dim=-1)
        return per_sample_loss.mean()

    def compute_loss(self, batch, ema_model=None, **kwargs):
        # Normalize input
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action']).to(self.device)

        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # Extract and normalize goal
        goal = None
        if self.use_goal_cond:
            if 'goal' not in batch:
                raise ValueError("Goal conditioning enabled but 'goal' not in batch")
            goal = self.normalizer['goal'].normalize(batch['goal']).to(self.device)
        raw_goal = batch.get('goal', None)
        if raw_goal is not None:
            raw_goal = raw_goal.to(self.device)

        remaining_episode_steps = batch.get('remaining_episode_steps', None)
        if remaining_episode_steps is not None:
            remaining_episode_steps = remaining_episode_steps.to(self.device)

        # Extract obstacles for collision loss
        obstacles = None
        if self.use_collision_loss:
            if 'obstacles' not in batch:
                cprint("[Warning] Collision loss enabled but no obstacles in batch", "yellow")
            else:
                obstacles = {
                    'cuboid_centers': batch['obstacles']['cuboid_centers'].to(self.device),
                    'cuboid_dims': batch['obstacles']['cuboid_dims'].to(self.device),
                    'cuboid_quaternions': batch['obstacles']['cuboid_quaternions'].to(self.device),
                }
        # Handle different ways of passing observation
        local_cond = None
        vis_cond = None
        trajectory = nactions
        cond_data = trajectory
        lang_cond = None
        ema_model = ema_model

        if self.language_conditioned:
            lang_cond = nobs.get('task_name', None)
            assert lang_cond is not None, "Language goal is required"

        # Encode observations.
        vis_cond = self._encode_obs(nobs, goal)

        # Split batch for flow and consistency training. Small tail batches can
        # otherwise make one branch empty and crash the attention reshape.
        flow_batchsize = int(batch_size * self.flow_batch_ratio)
        flow_batchsize = min(max(flow_batchsize, 1), batch_size)
        consistency_batchsize = batch_size - flow_batchsize

        # Get flow velocity targets
        flow_target_dict = self.get_flow_velocity(
            nactions[:flow_batchsize], 
            vis_cond=vis_cond[:flow_batchsize],
            lang_cond=lang_cond[:flow_batchsize] if lang_cond is not None else None
        )
        v_flow_pred = self.model(
            sample=flow_target_dict['x_t'], 
            timestep=flow_target_dict['t'].squeeze(),
            target_t=flow_target_dict['target_t'].squeeze(),
            vis_cond=vis_cond[:flow_batchsize],
            lang_cond=flow_target_dict['lang_cond'][:flow_batchsize] if lang_cond is not None else None
        )
        v_flow_pred_magnitude = torch.sqrt(torch.mean(v_flow_pred ** 2)).item()

        # Compute losses
        loss = torch.tensor(0.0, device=self.device)

        # Flow loss 
        v_flow_target = flow_target_dict['v_target']
        loss_flow = F.mse_loss(v_flow_pred, v_flow_target, reduction='none')
        loss_flow = reduce(loss_flow, 'b ... -> b (...)', 'mean')
        loss += loss_flow.mean()
        loss_flow = loss_flow.mean().item()

        # Consistency training loss
        loss_ct = torch.tensor(0.0, device=self.device)
        v_ct_pred_magnitude = 0.0
        if consistency_batchsize > 0:
            consistency_slice = slice(flow_batchsize, flow_batchsize + consistency_batchsize)
            consistency_target_dict = self.get_consistency_velocity(
                nactions[consistency_slice],
                vis_cond=vis_cond[consistency_slice],
                lang_cond=lang_cond[consistency_slice] if lang_cond is not None else None,
                ema_model=ema_model
            )
            v_ct_pred = self.model(
                sample=consistency_target_dict['x_t'], 
                timestep=consistency_target_dict['t'].squeeze(),
                target_t=consistency_target_dict['target_t'].squeeze(),
                vis_cond=vis_cond[consistency_slice],
                lang_cond=lang_cond[consistency_slice] if lang_cond is not None else None,
            )
            v_ct_pred_magnitude = torch.sqrt(torch.mean(v_ct_pred ** 2)).item()

            v_ct_target = consistency_target_dict['v_target']
            loss_ct = F.mse_loss(v_ct_pred, v_ct_target, reduction='none')
            loss_ct = reduce(loss_ct, 'b ... -> b (...)', 'mean')
            loss += loss_ct.mean()

        # Collision Loss
        loss_collision = torch.tensor(0.0, device=self.device)
        loss_terminal_goal = torch.tensor(0.0, device=self.device)
        collision_geometry_scale = self._get_collision_geometry_scale()
        collision_loss_weight_effective = (
            self.collision_loss_weight * self.collision_loss_schedule_multiplier
        )
        flow_lang_cond = (
            lang_cond[:flow_batchsize]
            if (lang_cond is not None and isinstance(lang_cond, torch.Tensor))
            else lang_cond
        )
        needs_terminal_rollout = (
            (self.use_collision_loss and obstacles is not None)
            or self.use_terminal_goal_reaching_loss
        )
        actions_pred_flow = None
        if needs_terminal_rollout:
            actions_pred_flow = self._rollout_flow_state_to_actions(
                x_t=flow_target_dict['x_t'],
                t=flow_target_dict['t'],
                vis_cond=vis_cond[:flow_batchsize],
                lang_cond=flow_lang_cond,
                num_steps=self.collision_num_steps,
            )
        if self.use_collision_loss and obstacles is not None:
            loss_collision = self._compute_collision_loss_unified(
                x_t=flow_target_dict['x_t'],
                t=flow_target_dict['t'],
                vis_cond=vis_cond[:flow_batchsize],
                lang_cond=flow_lang_cond,
                cuboid_centers=obstacles['cuboid_centers'][:flow_batchsize],
                cuboid_dims=obstacles['cuboid_dims'][:flow_batchsize],
                cuboid_quaternions=obstacles['cuboid_quaternions'][:flow_batchsize],
                num_steps=self.collision_num_steps,
                actions_pred=actions_pred_flow,
            )
            loss += collision_loss_weight_effective * collision_geometry_scale * loss_collision
        if self.use_terminal_goal_reaching_loss:
            if remaining_episode_steps is None:
                raise ValueError(
                    "Terminal goal reaching loss requires `remaining_episode_steps` in batch."
                )
            loss_terminal_goal = self._compute_terminal_goal_reaching_loss(
                actions_pred=actions_pred_flow,
                goal=raw_goal[:flow_batchsize] if raw_goal is not None else None,
                remaining_episode_steps=remaining_episode_steps[:flow_batchsize],
            )
            loss += self.terminal_goal_reaching_loss_weight * loss_terminal_goal

        loss_collision_scaled = collision_geometry_scale * loss_collision
        loss_terminal_goal_weighted = (
            self.terminal_goal_reaching_loss_weight * loss_terminal_goal
        )

        loss = loss.mean()

        def to_item(x):
            if isinstance(x, torch.Tensor):
                return x.item()
            return float(x)
        
        loss_dict = {
            'loss_flow': to_item(loss_flow),
            'loss_ct': to_item(loss_ct.mean()),  
            'loss_collision': to_item(loss_collision_scaled),
            'loss_collision_raw': to_item(loss_collision),
            'loss_collision_scale': collision_geometry_scale,
            'collision_loss_weight_raw': to_item(self.collision_loss_weight),
            'collision_loss_schedule_multiplier': to_item(self.collision_loss_schedule_multiplier),
            'collision_loss_weight_effective': to_item(collision_loss_weight_effective),
            'loss_terminal_goal_reaching': to_item(loss_terminal_goal_weighted),
            'loss_terminal_goal_reaching_raw': to_item(loss_terminal_goal),
            'terminal_goal_reaching_loss_weight': to_item(self.terminal_goal_reaching_loss_weight),
            'v_flow_pred_magnitude': v_flow_pred_magnitude,
            'v_ct_pred_magnitude': v_ct_pred_magnitude,
            'bc_loss': to_item(loss),
        }

        return loss, loss_dict
    
    def forward(self, batch, ema_model=None):
        """
        Forward method for DDP compatibility.
        DDP requires forward() to be called for gradient synchronization.
        """
        return self.compute_loss(batch, ema_model)
