#!/bin/bash
#SBATCH --job-name=andre_test
# Limit running time to 5 minutes.
#SBATCH -t 4:00:00  # time requested in hour:minute:second
# Request 1GB or RAM
#SBATCH --gres=gpu:1
#SBATCH --mem=30G -C v100 
##SBATCH --mem=30G -C v100 --cpu-per-task=10
#SBTACH -p compsci-gpu
python3 run_Limericks.py --type no_story --sasved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt child
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt applaud
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt art
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt time
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt market
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt pride
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt dog
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt random
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt opera
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt smart
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt cheat
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt useful
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt war
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt water
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt sports
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt library
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt car
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt scary
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt park
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt funeral
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt doctor
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt philosophy
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt evil
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt exercise
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt light
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt body
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt world
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt gun
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt forest
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt working
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt disease
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt animal
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt hope
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt death
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt cunning
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt fire
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt school
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt rich
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt violent
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt weight
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt creativity
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt law
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt angry
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt monster
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt leader
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt boxing
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt flower
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt union
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt fall
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt blood
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt music
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt loss
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt color
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt impressive
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt noble
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt ball
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt planet
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt night
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt airplane
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt love
# python3 run_Limericks.py --type no_story --saved_directory new_final_testing_DTS_no_story --mode multi --diversity True --cuda 1 --word_embedding_coefficient 0.1 --search_space 25 --retain_space 30 --prompt home

