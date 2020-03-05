#!/bin/bash
#SBATCH --job-name=naive_male
#SBATCH -t 48:00:00  
#SBATCH --gres=gpu:1
#SBATCH --mem=300G --cpus-per-task=20
#SBATCH -p compsci-gpu 
##--constraint=v100
#SBATCH -a 0-100 --ntasks-per-node=1
srun python3 run_Limericks.py  -dir "Naive"  -w 0.1 -ser 25 -re 30 -g "male"