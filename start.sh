#!/bin/bash

#SBATCH --qos=normal
#SBATCH --mem=4G
#SBATCH -c 4 
#SBATCH --gres=gpu:1 
#SBATCH -p gpu
#SBATCH --output=./slurm_out/slurm_%j.log

JOB_ID=${SLURM_JOB_ID}
CKPT_DIR=/checkpoint/ama/$JOB_ID

echo $JOB_ID
touch $CKPT_DIR/DELAYPURGE

#python main.py --method "standard" --job_id $JOB_ID --ckpt_dir $CKPT_DIR --pretrain ""
python main.py --method "adv" --job_id $JOB_ID --ckpt_dir $CKPT_DIR --pretrain "" --arch "resnet8" --lr 0.1
#python main.py --method "adv" --job_id $JOB_ID --ckpt_dir $CKPT_DIR --pretrain "" --arch "c11" --lr 0.01 
