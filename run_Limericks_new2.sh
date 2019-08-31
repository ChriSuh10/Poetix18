#!/bin/bash
CUDA_VISIBLE_DEVICES="0" python3 run_Limericks_new2.py --search_space 10
CUDA_VISIBLE_DEVICES="0" python3 run_Limericks_new2.py --search_space 50
CUDA_VISIBLE_DEVICES="0" python3 run_Limericks_new2.py --search_space 100
CUDA_VISIBLE_DEVICES="0" python3 run_Limericks_new2.py --search_space 10  --prompt "love"
CUDA_VISIBLE_DEVICES="0" python3 run_Limericks_new2.py --search_space 50 --prompt "love"
CUDA_VISIBLE_DEVICES="0" python3 run_Limericks_new2.py --search_space 100 --prompt "love"