#!/bin/bash
#SBATCH --job-name=andre_test
# Limit running time to 5 minutes.
#SBATCH -t 48:00:00  # time requested in hour:minute:second
# Request 1GB or RAM
#SBATCH --gres=gpu:1
#SBATCH --mem=100G --cpus-per-task=20
#SBTACH -p compsci-gpu
#SBATCH -a 0-100 --ntasks-per-node=1
srun python3 run_Limericks.py -t "original" -dir "Jan_DTS_story" -m "multi" -div True -w 0.1 -ser 25 -re 30