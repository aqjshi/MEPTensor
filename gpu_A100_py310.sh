#!/bin/bash
#SBATCH -p A100  # Ensure only one partition is specified
#SBATCH -c 12
#SBATCH -t 00:20:00
#SBATCH --mem=16gb
#SBATCH --gres=gpu:1
#SBATCH -o /gpfs/fs2/scratch/qshi10/pythonProject/output.log  # Redirect output to output.log
#SBATCH -e /gpfs/fs2/scratch/qshi10/pythonProject/error.log  # Redirect error to error.log

# Unload any existing CUDA modules
module unload cuda

# Load Conda environment
source /scratch/qshi10/miniconda3/etc/profile.d/conda.sh

# Activate the Conda environment
conda activate myenv_py310


export TF_ENABLE_ONEDNN_OPTS=0
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda/lib64

# Load CUDA and cuDNN modules
module load cuda/11.4
module load cudnn/11.4-8.2.4.15



# Verify CUDA installation
nvcc --version

# Check GPU visibility
nvidia-smi

# Run your Python script
python /gpfs/fs2/scratch/qshi10/pythonProject/model_cnn_master.py tensor_electrostatic_dataset.csv 9 1