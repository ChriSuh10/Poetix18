#!/bin/bash
CUDA_VISIBLE_DEVICES="3" python3 run_Limericks.py --search_space 200 --retain_space 5 --prompt   "tiger" --embedding 0.0
CUDA_VISIBLE_DEVICES="3" python3 run_Limericks.py --search_space 200 --retain_space 5 --prompt  "library" --embedding 0.0
CUDA_VISIBLE_DEVICES="3" python3 run_Limericks.py --search_space 200 --retain_space 5 --prompt "fairy" --embedding 0.0
CUDA_VISIBLE_DEVICES="3" python3 run_Limericks.py --search_space 200 --retain_space 5 --prompt   "duke" --embedding 0.0
CUDA_VISIBLE_DEVICES="3" python3 run_Limericks.py --search_space 200 --retain_space 5 --prompt  "print" --embedding 0.0
CUDA_VISIBLE_DEVICES="3" python3 run_Limericks.py --search_space 200 --retain_space 5 --prompt "click" --embedding 0.0