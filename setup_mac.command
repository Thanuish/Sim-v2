#!/bin/bash
# One-time setup on macOS: installs Python 3.12 (via uv, no admin needed) and all
# packages into ./.venv. Double-click this file in Finder, or run: ./setup_mac.command
set -e
cd "$(dirname "$0")"
echo "== MuJoCo VR Teleop setup =="

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "-> Installing uv (Python manager) into ~/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "-> Creating .venv with Python 3.12"
uv venv --python 3.12 --allow-existing .venv
echo "-> Installing packages (mujoco, aiohttp, h5py, ...)"
uv pip install --python .venv/bin/python -r requirements.txt
if command -v nvidia-smi >/dev/null 2>&1; then     # Linux with an NVIDIA GPU: the Newton cloth engine
  echo "-> NVIDIA GPU found: installing the Newton cloth engine"
  uv pip install --python .venv/bin/python -r requirements-newton.txt || echo "Newton install failed: the GPU (Taichi) cloth is used instead"
fi

if [ ! -f assets/universal_robots_ur5e/ur5e.xml ]; then
  .venv/bin/python setup_assets.py
fi

echo "-> Checking install"
.venv/bin/python -c "import mujoco, aiohttp, h5py; from vrteleop import scene; m=scene.build_scene().model; print('mujoco', mujoco.__version__, '- scene OK:', m.nbody, 'bodies')"
echo
echo "Setup complete. Start the simulator with:  ./start.command   (or double-click it)"
