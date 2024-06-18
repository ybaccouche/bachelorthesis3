#!/bin/bash
#SBATCH --job-name=testing
#SBATCH --time=00:10:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=defq
#SBATCH --gres=gpu:1

# Load GPU drivers

## Enable the following two lines for DAS5
module load cuda12.3/toolkit
module load cuDNN/cuda12.3

source $HOME/.bashrc
conda activate

mkdir $HOME/experiments
cd $HOME/experiments

echo $$
mkdir o`echo $$`
cd o`echo $$`

python /home/ybe320/Thesis/bachelor-thesis3/transformer.py 

