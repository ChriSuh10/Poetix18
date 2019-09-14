#!/bin/bash
CUDA_VISIBLE_DEVICES="1" python3 run_Limericks_new2.py --search_space 500 --prompt "sea"
CUDA_VISIBLE_DEVICES="1" python3 run_Limericks_new2.py --search_space 500 --prompt "monster"
CUDA_VISIBLE_DEVICES="1" python3 run_Limericks_new2.py --search_space 500 --prompt "hope"
CUDA_VISIBLE_DEVICES="1" python3 run_Limericks_new2.py --search_space 500 --prompt "money"
CUDA_VISIBLE_DEVICES="1" python3 run_Limericks_new2.py --search_space 500 --prompt "lust"
CUDA_VISIBLE_DEVICES="1" python3 run_Limericks_new2.py --search_space 500 --prompt "sex"