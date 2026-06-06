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

### 3. Install cuRobo

ESCAPE uses cuRobo for motion planning, so install it in the same
`env_isaaclab` environment before running inference:

```bash
conda activate env_isaaclab

# Use CUDA 12.8 with nvcc available. On systems with CUDA installed under
# /usr/local/cuda-12.8, point build tools there.
export CUDA_HOME="/usr/local/cuda-12.8"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# Confirm that nvcc is visible before installing cuRobo.
which nvcc
nvcc --version

# Set this for cuRobo builds. Adjust if your experiment needs a different arch.
export TORCH_CUDA_ARCH_LIST="8.0+PTX"

python -m pip install -e "git+https://github.com/NVlabs/curobo.git@ebb71702f3f70e767f40fd8e050674af0288abe8#egg=nvidia-curobo" --no-build-isolation
```

If your system does not have CUDA 12.8 with `nvcc`, install CUDA into the conda
environment and point build tools to that toolkit instead:

```bash
conda install -c nvidia cuda-toolkit=12.8 -y

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$LD_LIBRARY_PATH"
```

If `conda install -c nvidia cuda-toolkit=12.8 -y` fails with
`CondaVerificationError` for `nsight-compute` or a `ClobberError` involving
CUDA Nsight packages, clear the corrupted conda package cache and retry:

```bash
# This removes downloaded conda package caches, not installed environments.
conda clean --packages -y
conda install -c nvidia cuda-toolkit=12.8 -y
```

### 4. Install PyTorch3D and Flash Attention

ESCAPE requires PyTorch3D and Flash Attention. Install them in the same
`env_isaaclab` environment before running inference.

```bash
# Make sure you are using the ESCAPE / Isaac Lab environment.
conda activate env_isaaclab

# Query your GPU compute capability.
python -c "import torch; print(torch.cuda.get_device_capability())"
```

Example outputs:

```bash
# RTX 4090D example
(env_isaaclab) amdin@amdin-Z790-EAGLE-AX:~/ESCAPE_release$ python -c "import torch; print(torch.cuda.get_device_capability())"
(8, 9)

# RTX 5090 example
(env_isaaclab) amdin@amdin-Z790-EAGLE-AX:~/ESCAPE_release$ python -c "import torch; print(torch.cuda.get_device_capability())"
(12, 0)
```

Set `TORCH_CUDA_ARCH_LIST` from the output above by joining the two numbers with
a dot. For example, `(8, 9)` becomes `8.9`, and `(12, 0)` becomes `12.0`.
For multiple GPUs, separate values with semicolons, such as `"8.9;12.0"`.

```bash
# Choose the value that matches your GPU.
export TORCH_CUDA_ARCH_LIST="8.9"   # RTX 4090D
# export TORCH_CUDA_ARCH_LIST="12.0"  # RTX 5090

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

Successful inference example:

<video src="docs/assets/ESCAPE-example.mp4" controls width="100%">
Your browser does not support the video tag.
</video>

Video: [ESCAPE-example.mp4](docs/assets/ESCAPE-example.mp4)

Optional arguments:

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
