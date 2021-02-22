#!/bin/bash

#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH -n 14

#SBATCH -t 48:00:00
#SBATCH --mem=100GB

module purge
source ~/.bashrc
source activate pytorch-1.4

python ./train_bert.py
