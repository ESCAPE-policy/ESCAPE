# ESCAPE Pre-release

This repository contains the pre-release version of ESCAPE, with the inference
code, model components, checkpoints, robot assets, Isaac Lab task registration,
and runtime scripts needed to run checkpoint-based motion planning.

`Note:` This pre-release focuses on inference. We will be adding training code,
datasets, and more examples for the full release. Stay tuned!


## Installation

### Requirements

- Ubuntu 22.04 LTS
- NVIDIA GPU with at least 24 GB VRAM, such as RTX 4090D or better
- NVIDIA driver >= 560.35.05 and CUDA >= 12.6
- Anaconda or Miniconda

### 1. Base Environment and Isaac Lab

ESCAPE is tested with Isaac Lab on Isaac Sim 5.1.0. Create a Python 3.11 conda
environment and install the Isaac Lab stack first:

```bash
conda create -n env_isaaclab python=3.11 -y
conda activate env_isaaclab
pip install --upgrade pip

pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com

sudo apt install cmake build-essential git libvulkan1 mesa-vulkan-drivers vulkan-tools -y

cd ~
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install
```

Verify the Isaac Lab installation:

```bash
./isaaclab.sh -p scripts/environments/zero_agent.py --task Isaac-Cartpole-Direct-v0 --num_envs 128
```

### 2. Install ESCAPE

Install ESCAPE in the same `env_isaaclab` environment:

```bash
# Clone and install ESCAPE
conda activate env_isaaclab
cd ~
git clone git@github.com:ESCAPE-policy/ESCAPE.git
cd ESCAPE

# Install ESCAPE runtime dependencies
pip install -r requirements.txt

# Install ESCAPE as an editable Isaac Lab extension
python -m pip install -e source/ESCAPE
```

If your Isaac Lab install is not on the active Python environment, use the
Isaac Lab launcher Python instead:

```bash
<PATH_TO_ISAACLAB>/isaaclab.sh -p -m pip install -r requirements.txt
<PATH_TO_ISAACLAB>/isaaclab.sh -p -m pip install -e source/ESCAPE
```

### 3. Optional: PyTorch3D and Flash Attention

Some checkpoints or downstream experiments may use PyTorch3D or Flash Attention.
Install them in the same environment if needed:

```bash
# Query GPU compute capability
python -c "import torch; print(torch.cuda.get_device_capability())"

# RTX 4090D uses 8.9. For multiple GPUs, separate values with semicolons.
export TORCH_CUDA_ARCH_LIST="8.9"
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Optional, may take several minutes to compile
MAX_JOBS=4 python -m pip -v install flash-attn --no-build-isolation
```

## Test the Installation

Run the example inference script with one of the public scenes:

```bash
python scripts/motion_planning/ESCAPE_inference.py \
  --checkpoint_path /path/to/checkpoint.ckpt \
  --scene_name scene_3
```

Optional arguments:

- `--demo_id 0`
- `--headless`

Public scenes are `scene_1`, `scene_2`, and `scene_3`. Each scene exposes
`demo_0` with five target trajectories, for a total of 15 public trajectories.
The target metadata is stored in
`assets/scenes/targets/escape_public_targets.hdf5`.

## Repository Contents

- `scripts/motion_planning/`: checkpoint inference entry points.
- `source/ESCAPE/ESCAPE/maniflow/`: policy and model code required to
  instantiate checkpoints.
- `source/ESCAPE/ESCAPE/motion_planners/`: ManiFlow and CuRobo planner wrappers.
- `source/ESCAPE/ESCAPE/tasks/manager_based/`: Isaac Lab task registrations used
  by inference.
- `assets/`, `robofin/`: robot assets, target metadata, and geometry utilities
  needed at runtime.

## Citation

Citation information will be added in the full release.
