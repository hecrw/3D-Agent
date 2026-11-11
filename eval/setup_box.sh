#!/usr/bin/env bash
# One-time setup of the GPU box for eval/rerun_all.sh — no sudo needed.
# Everything goes under $HOME (Miniforge, conda envs, model repos).
#
# Run the stages in order, from the repo root (~/3D-Agent):
#   bash eval/setup_box.sh conda        # installs Miniforge, then OPEN A NEW SHELL
#   bash eval/setup_box.sh buildtools   # CUDA 12.4 nvcc + gcc 12 (Ubuntu's gcc 15 is too new for nvcc 12.4)
#   bash eval/setup_box.sh trellis      # TRELLIS.2 env "trellis" (slowest: flash-attn build, ~30-60 min)
#   bash eval/setup_box.sh hunyuan      # Hunyuan3D-2 env "hunyuan"
#   bash eval/setup_box.sh partcrafter  # PartCrafter env "partcrafter"
#   bash eval/setup_box.sh ulip         # ULIP-2 env "ulip" (ULIP repo + weights auto-download on first score)
#   bash eval/setup_box.sh modal        # log the eval venv into Modal
# Then:  bash eval/rerun_all.sh preflight
#
# trellis / hunyuan / partcrafter / ulip are independent: after buildtools you
# can run them in separate terminals at the same time.
set -uo pipefail

CONDA_DIR="${CONDA_DIR:-$HOME/miniforge3}"
TOOLS_ENV="cuda124"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"   # RTX A6000 = sm_86
export MAX_JOBS="${MAX_JOBS:-8}"                            # parallel compile jobs (shared box: keep moderate)
CU124="https://download.pytorch.org/whl/cu124"

load_conda() {
  [ -f "$CONDA_DIR/etc/profile.d/conda.sh" ] || { echo "conda not installed — run: bash eval/setup_box.sh conda"; exit 1; }
  # shellcheck disable=SC1091
  . "$CONDA_DIR/etc/profile.d/conda.sh"
}

# Point every CUDA extension build at the conda nvcc 12.4 + gcc 12 toolchain.
use_buildtools() {
  local T="$CONDA_DIR/envs/$TOOLS_ENV"
  [ -x "$T/bin/nvcc" ] || { echo "no nvcc — run: bash eval/setup_box.sh buildtools"; exit 1; }
  export CUDA_HOME="$T" CUDA_PATH="$T"
  export PATH="$T/bin:$PATH"
  export CC="$T/bin/x86_64-conda-linux-gnu-gcc" CXX="$T/bin/x86_64-conda-linux-gnu-g++"
  export NVCC_PREPEND_FLAGS="-ccbin $CXX"
  export CPATH="$T/targets/x86_64-linux/include:$T/include:${CPATH:-}"
  export LIBRARY_PATH="$T/targets/x86_64-linux/lib:$T/lib:${LIBRARY_PATH:-}"
  export LD_LIBRARY_PATH="$T/targets/x86_64-linux/lib:$T/lib:${LD_LIBRARY_PATH:-}"
  echo "toolchain: $(nvcc --version | tail -1) | $($CXX --version | head -1)"
}

new_env() {  # new_env <name> <python>
  conda env list | awk '{print $1}' | grep -qx "$1" || conda create -y -n "$1" "python=$2"
  conda activate "$1"
  python -m pip install -U pip wheel setuptools ninja
}

clone() {  # clone <url> <dir>
  [ -d "$2" ] || git clone --recursive "$1" "$2"
}

case "${1:-}" in
conda)
  if [ -d "$CONDA_DIR" ]; then echo "Miniforge already at $CONDA_DIR"; else
    f=/tmp/miniforge-$$.sh
    curl -fsSL -o "$f" "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    bash "$f" -b -p "$CONDA_DIR" && rm -f "$f"
  fi
  "$CONDA_DIR/bin/conda" init bash
  "$CONDA_DIR/bin/conda" config --set auto_activate_base false
  echo; echo ">>> Now open a NEW shell (or: source ~/.bashrc), then run the buildtools stage."
  ;;
buildtools)
  load_conda
  conda env list | awk '{print $1}' | grep -qx "$TOOLS_ENV" || \
    conda create -y -n "$TOOLS_ENV" -c "nvidia/label/cuda-12.4.1" -c conda-forge \
      cuda-toolkit "gcc_linux-64=12" "gxx_linux-64=12"
  use_buildtools
  ;;
trellis)
  load_conda; use_buildtools
  clone https://github.com/microsoft/TRELLIS.2.git "$HOME/TRELLIS.2"
  new_env trellis 3.10
  pip install torch==2.6.0 torchvision==0.21.0 --index-url "$CU124"
  pip install fastapi uvicorn python-multipart
  cd "$HOME/TRELLIS.2"
  # setup.sh installs into the ACTIVE env when --new-env is not given.
  # shellcheck disable=SC1091
  . ./setup.sh --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm
  python -c "import torch, trellis2, o_voxel; print('trellis OK', torch.__version__, torch.cuda.is_available())"
  ;;
hunyuan)
  load_conda; use_buildtools
  clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git "$HOME/Hunyuan3D-2"
  new_env hunyuan 3.10
  pip install torch==2.5.1 torchvision==0.20.1 --index-url "$CU124"
  pip install diffusers==0.32.2 transformers==4.49.0 accelerate einops omegaconf \
      trimesh pymeshlab pygltflib xatlas opencv-python-headless 'numpy<2' \
      'tqdm>=4.66.3' rembg onnxruntime-gpu 'huggingface_hub[hf_transfer]'
  cd "$HOME/Hunyuan3D-2"
  pip install -e . --no-deps
  (cd hy3dgen/texgen/custom_rasterizer && pip install . --no-build-isolation)
  (cd hy3dgen/texgen/differentiable_renderer && pip install . --no-build-isolation)
  python -c "import torch, hy3dgen, custom_rasterizer; print('hunyuan OK', torch.__version__, torch.cuda.is_available())"
  ;;
partcrafter)
  load_conda
  clone https://github.com/wgsxm/PartCrafter.git "$HOME/PartCrafter"
  new_env partcrafter 3.11
  pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url "$CU124"
  pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
  pip install scikit-learn diffusers transformers einops 'huggingface_hub[hf_transfer]' \
      opencv-python-headless trimesh omegaconf scikit-image numpy==1.26.4 peft \
      jaxtyping typeguard matplotlib imageio imageio-ffmpeg pyrender colormaps \
      accelerate pillow
  python -c "import torch, torch_cluster; print('partcrafter OK', torch.__version__, torch.cuda.is_available())"
  ;;
ulip)
  load_conda
  new_env ulip 3.10
  pip install torch==2.5.1 torchvision==0.20.1 --index-url "$CU124"
  pip install open_clip_torch==2.24.0 timm easydict pyyaml numpy 'huggingface_hub[hf_transfer]'
  python -c "import torch, open_clip; print('ulip OK', torch.__version__, torch.cuda.is_available())"
  ;;
modal)
  echo "Log in as the account in .env's TRELLIS_WORKSPACE (the one PartCrafter/Hunyuan3D-2 are deployed to)."
  .venv/bin/modal token new
  .venv/bin/modal profile current
  ;;
*) sed -n 2,18p "$0"; exit 1 ;;
esac
