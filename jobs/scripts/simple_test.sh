#!/bin/bash
#SBATCH --job-name=simple_test
#SBATCH --time=00:10:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=/home/ybe320/Thesis/bachelor-thesis/jobs/outputs/simple_job_%j.out
#SBATCH --error=/home/ybe320/Thesis/bachelor-thesis/jobs/outputs/simple_job_%j.err

echo "This is a test job"
echo "SLURM job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Current directory: $(pwd)"
date