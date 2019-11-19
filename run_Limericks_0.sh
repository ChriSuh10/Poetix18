#!/bin/bash
CUDA_VISIBLE_DEVICES="0" python3 run_Limericks.py --search_space 100 --retain_space 5 --prompt  "hound" --embedding 0.0
CUDA_VISIBLE_DEVICES="0" python3 run_Limericks.py --search_space 100 --retain_space 5 --prompt  "blood" --embedding 0.0
CUDA_VISIBLE_DEVICES="0" python3 run_Limericks.py --search_space 100 --retain_space 5 --prompt  "death" --embedding 0.0
CUDA_VISIBLE_DEVICES="0" python3 run_Limericks.py --search_space 100 --retain_space 5 --prompt   "war" --embedding 0.0
CUDA_VISIBLE_DEVICES="2" python3 run_Limericks.py --search_space 100 --retain_space 5 --prompt  "queen" --embedding 0.0
