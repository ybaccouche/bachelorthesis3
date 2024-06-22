#!/bin/bash
#SBATCH --job-name=work_pls                      # Job name
#SBATCH --time=05:00:00                     # Time limit hrs:min:sec
#SBATCH -N 1                                 # Number of nodes
#SBATCH --ntasks-per-node=1                  # Number of tasks per node
#SBATCH --partition=defq   # Partition name
#SBATCH --gres=gpu:1                         # Number of GPUs per node
#SBATCH --output=/home/ybe320/Thesis/bachelor-thesis3/bachelorthesis3/jobs/outputs/job_%j.out  # Output file
#SBATCH --error=/home/ybe320/Thesis/bachelor-thesis3/bachelorthesis3/jobs/outputs/job_%j.err   # Error file

# Uncomment and set the correct CUDA module if required by your cluster setup
module load cuda12.1/toolkit
module load cuDNN/cuda12.1

# Activate Anaconda environment
source $HOME/.bashrc
conda activate

# Navigate to the existing experiment directory
BASE_DIR=$HOME/Thesis/bachelor-thesis3/bachelorthesis3/
cd $BASE_DIR

# Run the Python script
python /home/ybe320/Thesis/bachelor-thesis3/bachelorthesis3/transformer.py --batch_size 32 --epochs 50000 --lr 0.001 --heads 8 --depth 12
