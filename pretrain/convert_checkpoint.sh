#!/bin/bash

#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH -n 14 
#SBATCH -t 48:00:00
#SBATCH --mem=100GB

# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err

module purge
source ~/.bashrc
#source activate tensorflow-1.15
source activate pytorch-1.4

export BERT_BASE_DIR=/scratch/nx296/kaggle/news/pretrain
python convert_checkpoint.py \
  --tf_checkpoint_path $BERT_BASE_DIR/output/model.ckpt-315000 \
  --bert_config_file ./config.json \
  --pytorch_dump_path ./pytorch_model.bin

