#!/bin/bash -x
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=128G
###SBATCH --cpus-per-task=4
#SBATCH -t 1-00:00              # time limit: (D-HH:MM) 
#SBATCH --job-name=viper_chatgpt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=piyushkh@andrew.cmu.edu


module load cuda-11.8

export CUDA_HOME=/usr/local/cuda-11.8

source /data/tir/projects/tir6/general/piyushkh/conda/bin/activate vipergpt2

SAFETENSORS_FAST_GPU=1 python main_batch.py > viper_visual_search_final2.txt

# > gpt3-5_v_star_test.txt