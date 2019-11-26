#!/bin/bash
CUDA_VISIBLE_DEVICES="2" python3 run_Limericks.py --search_space 200 --retain_space 5 --prompt  "game" --embedding 0.0
CUDA_VISIBLE_DEVICES="2" python3 run_Limericks.py --search_space 200 --retain_space 5 --prompt "love" --embedding 0.0
CUDA_VISIBLE_DEVICES="2" python3 run_Limericks.py --search_space 200 --retain_space 5 --prompt  "vegetable" --embedding 0.0
CUDA_VISIBLE_DEVICES="2" python3 run_Limericks.py --search_space 200 --retain_space 5 --prompt  "fish" --embedding 0.0
CUDA_VISIBLE_DEVICES="2" python3 run_Limericks.py --search_space 200 --retain_space 5 --prompt  "theater" --embedding 0.0