#!/bin/bash
project="wandb_project_name"

for lr in 0.01; do
	bash launch_slurm_job.sh gpu job_${lr} 1 "python3 main.py --method \"standard\" --lr ${lr} --epoch 2 --wandb_project \"${project}\""
done
