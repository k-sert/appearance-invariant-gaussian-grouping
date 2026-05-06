#!/bin/bash
set -e

module load 2025
module load Miniconda3/25.5.1-1
module load CUDA/12.8.0

source $(conda info --base)/etc/profile.d/conda.sh

ENV_PATH="$HOME/envs/gg-env"

conda create -p "$ENV_PATH" python=3.8 -y
conda activate "$ENV_PATH"

pip install torch==2.4.1+cu124 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu124

pip install plyfile==0.8.1 tqdm scipy wandb opencv-python scikit-learn lpips ninja codecarbon

export CUDA_HOME=$EBROOTCUDA
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="8.0"

cd "$HOME/Appearance-Invariant-Gaussian-Grouping/Stage-1/Gaussian-Grouping"

grep -q "#include <cfloat>" ./submodules/simple-knn/simple_knn.cu || \
sed -i '1i #include <cfloat>' ./submodules/simple-knn/simple_knn.cu

pip install -v --no-build-isolation ./submodules/diff-gaussian-rasterization
pip install -v --no-build-isolation ./submodules/simple-knn

python - <<'EOF'
import torch
import diff_gaussian_rasterization
import simple_knn
print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("✅ GG environment setup complete")
EOF
