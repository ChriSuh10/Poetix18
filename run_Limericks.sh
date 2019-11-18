#!/bin/bash
CUDA_VISIBLE_DEVICES="2" python3 run_Limericks.py --search_space 100 --retain_space 3 --prompt  "planet" --embedding 0.0
CUDA_VISIBLE_DEVICES="2" python3 run_Limericks.py --search_space 100 --retain_space 3 --prompt  "planet" --embedding 0.1
CUDA_VISIBLE_DEVICES="2" python3 run_Limericks.py --search_space 100 --retain_space 3 --prompt  "planet" --embedding 0.2
CUDA_VISIBLE_DEVICES="2" python3 run_Limericks.py --search_space 100 --retain_space 3 --prompt   "planet" --embedding 0.3
CUDA_VISIBLE_DEVICES="2" python3 run_Limericks.py --search_space 100 --retain_space 3 --prompt  "planet" --embedding 0.4
CUDA_VISIBLE_DEVICES="2" python3 run_Limericks.py --search_space 100 --retain_space 3 --prompt "planet" --embedding 0.5