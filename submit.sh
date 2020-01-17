#!/bin/bash
#SBATCH --job-name=andre_test
# Limit running time to 5 minutes.
#SBATCH -t 48:00:00  # time requested in hour:minute:second
# Request 1GB or RAM
#SBATCH --gres=gpu:1
#SBATCH --mem=100G --cpus-per-task=20
#SBTACH -p compsci-gpu
#SBATCH -a 0-100 --ntasks-per-node=1
srun python3 run_Limericks.py --type no_story --saved_directory Jan_DTS_no_story --mode multi --diversity True --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30