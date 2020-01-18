#!/bin/bash
#SBATCH --job-name=story
#SBATCH -t 48:00:00  
#SBATCH --gres=gpu:1
#SBATCH --mem=300G --cpus-per-task=20
#SBTACH -p compsci-gpu --constraint=v100 --exclude=linux[1-50],gpu-compute[1-4]
#SBATCH -a 0-19 --ntasks-per-node=1
srun python3 run_Limericks.py -t "original" -dir "Jan_DTS_story" -m "multi" -div True -w 0.1 -ser 25 -re 30