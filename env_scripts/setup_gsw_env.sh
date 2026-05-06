#!/bin/bash
set -e

echo "[1/7] Loading Snellius modules..."

module load 2025
module load Miniconda3/25.5.1-1
module load CUDA/12.8.0

echo "[2/7] Initializing conda..."

source "$(conda info --base)/etc/profile.d/conda.sh"

ENV_PATH="$HOME/envs/gsw-env"

echo "[3/7] Creating conda environment..."

conda create -y -p "$ENV_PATH" python=3.10 pip ninja cmake
conda activate "$ENV_PATH"

echo "[4/7] Installing PyTorch..."

pip install --upgrade pip setuptools wheel

pip install \
  torch==2.4.1 \
  torchvision==0.19.1 \
  torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu124

echo "[5/7] Installing Python dependencies..."

pip install \
  plyfile \
  tqdm \
  scipy \
  pandas \
  pillow \
  opencv-python \
  imageio \
  imageio-ffmpeg \
  matplotlib \
  scikit-image \
  scikit-learn \
  lpips \
  tensorboard \
  einops \
  kornia \
  codecarbon

echo "[6/7] Configuring CUDA environment..."

export CUDA_HOME="${EBROOTCUDA}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="8.0"

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. && pwd )"
cd "$REPO_ROOT/Stage-1/Gaussian-in-the-Wild"

grep -q "#include <cfloat>" ./submodules/simple-knn/simple_knn.cu || \
  sed -i '1i #include <cfloat>' ./submodules/simple-knn/simple_knn.cu

echo "[7/7] Building CUDA extensions..."

pip install -v --no-build-isolation ./submodules/diff-gaussian-rasterization
pip install -v --no-build-isolation ./submodules/simple-knn

python - <<'EOF'
import torch
import diff_gaussian_rasterization
import simple_knn

print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

print("✅ CUDA extensions imported successfully")
print("✅ GS-W environment setup complete")
EOF