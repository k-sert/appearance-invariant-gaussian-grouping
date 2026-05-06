#!/bin/bash

module load 2025
module load Miniconda3/25.5.1-1
module load CUDA/12.8.0

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ~/envs/gg-env

export CUDA_HOME=$EBROOTCUDA
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="8.0"

echo "✅ GG environment activated"
echo "Python: $(which python)"
echo "CUDA_HOME: $CUDA_HOME"
