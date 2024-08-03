#!/bin/bash
#SBATCH -p h100  # Ensure only one partition is specified
#SBATCH -c 12
#SBATCH -t 00:20:00
#SBATCH --mem=16gb
#SBATCH --gres=gpu:1
#SBATCH -o /gpfs/fs2/scratch/qshi10/pythonProject/output.log  # Redirect output to output.log
#SBATCH -e /gpfs/fs2/scratch/qshi10/pythonProject/error.log  # Redirect error to error.log

# Load Anaconda module (if needed by your system)
module load anaconda3

# Initialize Conda for bash shell
source /scratch/qshi10/anaconda3/etc/profile.d/conda.sh

# Activate the Conda environment
conda activate tf_py37

# Check CUDA version
nvcc --version

# Check GPU visibility
nvidia-smi

# Run your Python script
python /gpfs/fs2/scratch/qshi10/pythonProject/gpu.py
