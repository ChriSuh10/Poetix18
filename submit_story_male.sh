#!/bin/bash
#SBATCH --job-name=multi_male
#SBATCH -t 48:00:00  
#SBATCH --gres=gpu:1
#SBATCH --mem=300G --cpus-per-task=20
#SBATCH -p compsci-gpu 
##--constraint=v100
#SBATCH -a 0-4 --ntasks-per-node=1
srun python3 run_Limericks.py -t "original" -dir "Andre_story" -m "multi" -div True -w 0.1 -ser 25 -re 30 -g "male"