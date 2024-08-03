#!/bin/bash
#SBATCH -p standard  # Ensure only one partition is specified
#SBATCH -c 8
#SBATCH -t 00:20:00
#SBATCH --mem=8gb

# Unload any existing CUDA modules
module unload cuda

# Load Conda environment
source /scratch/qshi10/anaconda3/etc/profile.d/conda.sh || { echo "Failed to source conda.sh"; exit 1; }

# Activate the Conda environment
conda activate tf_py37 || { echo "Failed to activate conda environment"; exit 1; }

# Set environment variables for CUDA and cuDNN
export PATH=/scratch/qshi10/anaconda3/envs/tf_py37/bin:$PATH
export LD_LIBRARY_PATH=/scratch/qshi10/anaconda3/envs/tf_py37/lib:$LD_LIBRARY_PATH

# Verify CUDA installation
nvcc --version || { echo "nvcc not found"; exit 1; }

# Check GPU visibility
nvidia-smi || { echo "nvidia-smi command failed"; exit 1; }