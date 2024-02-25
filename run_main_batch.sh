#!/bin/bash -x
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=128G
###SBATCH --cpus-per-task=4
#SBATCH -t 2-00:00              # time limit: (D-HH:MM) 
#SBATCH --job-name=viper_OkVQA_all
#SBATCH --mail-type=ALL
#SBATCH --mail-user=piyushkh@andrew.cmu.edu
#SBATCH --partition=babel-shared-long

module load cuda-11.8

export CUDA_HOME=/usr/local/cuda-11.8

source /data/tir/projects/tir6/general/piyushkh/conda/bin/activate vipergpt2

python main_batch.py > main_batch_output.txt