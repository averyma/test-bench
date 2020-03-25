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
#python main.py --method "adv" --job_id $JOB_ID --ckpt_dir $CKPT_DIR --pretrain ""
#python main.py --method "soar" --job_id $JOB_ID --ckpt_dir $CKPT_DIR --lr_update 90 100 --epoch 150 --lr 0.002
python main.py --method "exp" --job_id $JOB_ID --ckpt_dir $CKPT_DIR 
