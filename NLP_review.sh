#!/bin/bash
#SBATCH --job-name=pacifier
#SBATCH -t 48:00:00  
#SBATCH --gres=gpu:1
#SBATCH --mem=300G --cpus-per-task=20
#SBATCH -p compsci-gpu 
##--constraint=v100
#SBATCH -a 0 --ntasks-per-node=1
srun python3 NLP_for_review.py -p "pacifier"