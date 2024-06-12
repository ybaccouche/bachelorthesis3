#!/bin/bash
#SBATCH --job-name=long_job
#SBATCH --time=06:00:00  # Set to 6 hours
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=/home/ybe320/Thesis/bachelor-thesis/jobs/outputs/long_job_%j.out
#SBATCH --error=/home/ybe320/Thesis/bachelor-thesis/jobs/outputs/long_job_%j.err

echo "Starting job script"
date

# Move to the correct directory
cd /home/ybe320/Thesis/bachelor-thesis/
if [ $? -ne 0 ]; then
  echo "Failed to change directory to /home/ybe320/Thesis/bachelor-thesis/"
  exit 1
fi

echo "SLURM job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Current directory: $(pwd)"
date

# Load GPU drivers
module load cuda12.3/toolkit
if [ $? -ne 0 ]; then
  echo "Failed to load CUDA toolkit"
  exit 1
fi

# Source bashrc and activate conda environment
source $HOME/.bashrc
if [ $? -ne 0 ]; then
  echo "Failed to source .bashrc"
  exit 1
fi

conda activate
if [ $? -ne 0 ]; then
  echo "Failed to activate conda environment"
  exit 1
fi

# Install required packages from requirements.txt
pip install -r requirements.txt
if [ $? -ne 0 ]; then
  echo "Failed to install required packages"
  exit 1
fi

# Ensure wget is installed
pip install wget
if [ $? -ne 0 ]; then
  echo "Failed to install wget module"
  exit 1
fi

# Set wandb API key
export WANDB_API_KEY=694fb34144778f8ff751adfb317024489e1a077e

# Base directory for the experiment
mkdir -p experiments
if [ $? -ne 0 ]; then
  echo "Failed to create experiments directory"
  exit 1
fi

cd experiments
if [ $? -ne 0 ]; then
  echo "Failed to change directory to experiments"
  exit 1
fi

# Create a unique directory for this run
run_dir="o$(date +%s)"
mkdir $run_dir
if [ $? -ne 0 ]; then
  echo "Failed to create run directory"
  exit 1
fi

cd $run_dir
if [ $? -ne 0 ]; then
  echo "Failed to change directory to run directory"
  exit 1
fi

# Debugging output
echo "Current directory after mkdir and cd: $(pwd)"
echo "Conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"
date

# Run a simple Python test
python <<EOF
import torch
print("CUDA available:", torch.cuda.is_available())
EOF
if [ $? -ne 0 ]; then
  echo "Python test failed"
  exit 1
fi

# Run transformer.py
if [ -f "/home/ybe320/Thesis/bachelor-thesis/transformer.py" ]; then
    python /home/ybe320/Thesis/bachelor-thesis/transformer.py --batch_size 32
    if [ $? -ne 0 ]; then
        echo "transformer.py execution failed"
        exit 1
    fi
else
    echo "transformer.py not found"
    exit 1
fi

# More debugging output
echo "Long job script completed"
date