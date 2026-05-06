#!/bin/bash

echo "[1/4] Loading Snellius modules..."

module load 2025
module load Miniconda3/25.5.1-1
module load CUDA/12.8.0

echo "[2/4] Initializing conda..."

source "$(conda info --base)/etc/profile.d/conda.sh"

echo "[3/4] Activating GS-W environment..."

conda activate "$HOME/envs/gsw-env"

echo "[4/4] Configuring CUDA environment..."

export CUDA_HOME="${EBROOTCUDA}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="8.0"

echo "✅ GS-W environment activated"
echo "Python: $(which python)"
echo "CUDA_HOME: $CUDA_HOME"
echo "TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"