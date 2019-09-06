#!/bin/bash
CUDA_VISIBLE_DEVICES="1" python3 run_Limericks_new2.py --search_space 50 --prompt "sea"
CUDA_VISIBLE_DEVICES="1" python3 run_Limericks_new2.py --search_space 50 --prompt "monster"
CUDA_VISIBLE_DEVICES="1" python3 run_Limericks_new2.py --search_space 50 --prompt "hope"
CUDA_VISIBLE_DEVICES="1" python3 run_Limericks_new2.py --search_space 50 --prompt "money"
CUDA_VISIBLE_DEVICES="1" python3 run_Limericks_new2.py --search_space 50 --prompt "lust"
CUDA_VISIBLE_DEVICES="1" python3 run_Limericks_new2.py --search_space 50 --prompt "sex"