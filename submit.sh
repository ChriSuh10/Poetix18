#!/bin/bash
#SBATCH --job-name=story
# Limit running time to 5 minutes.
#SBATCH -t 48:00:00  # time requested in hour:minute:second
# Request 1GB or RAM
#SBATCH --gres=gpu:1
#SBATCH --mem=300G --cpus-per-task=20
#SBTACH -p compsci-gpu --constraint=v100 --nodelist=gpu-compute[5-7]
#SBATCH -a 0-19 --ntasks-per-node=1
srun python3 run_Limericks.py -t "original" -dir "Jan_DTS_story" -m "multi" -div True -w 0.1 -ser 25 -re 30