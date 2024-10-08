#!/bin/bash
#SBATCH --job-name=final_pr                    # Job name
#SBATCH --time=36:00:00                     # Time limit hrs:min:sec
#SBATCH -N 3                             # Number of nodes
#SBATCH --ntasks-per-node=1                  # Number of tasks per node
#SBATCH --partition=defq   # Partition name
#SBATCH --gres=gpu:2                        # Number of GPUs per node
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
python /home/ybe320/Thesis/bachelor-thesis3/bachelorthesis3/ev_peceiver.py #--batch_size 32 --num_batches 1000 #--lr 0.001 --nblocks 8 --nheads 12
