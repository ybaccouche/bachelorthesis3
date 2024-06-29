#!/bin/bash
#SBATCH --job-name=final_pr                    # Job name
#SBATCH --time=24:00:00                     # Time limit hrs:min:sec
#SBATCH -N 1                             # Number of nodes
#SBATCH --ntasks-per-node=1                  # Number of tasks per node
#SBATCH --partition=gpu   # Partition name
#SBATCH --gres=gpu:1                        # Number of GPUs per node
#SBATCH --output=/home/scur2273/bachelorthesis3/jobs/outputs/job_%j.out  # Output file  # Output file
#SBATCH --error=/home/scur2273/bachelorthesis3/jobs/outputs/job_%j.err   # Error file


# Uncomment and set the correct CUDA module if required by your cluster setup
module purge
module load 2022
module load Anaconda3/2022.05

# Activate Anaconda environment
source $HOME/.bashrc
conda activate ybe

# Run the Python script
python /home/scur2273/bachelorthesis3/ev_perceiver.py #--batch_size 32 --num_batches 1000 #--lr 0.001 --nblocks 8 --nheads 12
