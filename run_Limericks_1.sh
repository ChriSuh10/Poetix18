#!/bin/bash
CUDA_VISIBLE_DEVICES="1" python3 run_Limericks.py --search_space 200 --retain_space 5 --prompt "happy" --embedding 0.0
CUDA_VISIBLE_DEVICES="1" python3 run_Limericks.py --search_space 200 --retain_space 5 --prompt  "world" --embedding 0.0
CUDA_VISIBLE_DEVICES="1" python3 run_Limericks.py --search_space 200 --retain_space 5 --prompt  "planet" --embedding 0.0
CUDA_VISIBLE_DEVICES="1" python3 run_Limericks.py --search_space 200 --retain_space 5 --prompt  "fire" --embedding 0.0
CUDA_VISIBLE_DEVICES="1" python3 run_Limericks.py --search_space 200 --retain_space 5 --prompt   "water" --embedding 0.0