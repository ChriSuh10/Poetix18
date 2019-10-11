#!/bin/bash
CUDA_VISIBLE_DEVICES="2" python3 run_Limericks_new2.py --search_space 50 --retain_space 5 --prompt "fish"
CUDA_VISIBLE_DEVICES="2" python3 run_Limericks_new2.py --search_space 50 --retain_space 5 --prompt "fire"
CUDA_VISIBLE_DEVICES="2" python3 run_Limericks_new2.py --search_space 50 --retain_space 5 --prompt "trip"
CUDA_VISIBLE_DEVICES="2" python3 run_Limericks_new2.py --search_space 50 --retain_space 5 --prompt "water"
CUDA_VISIBLE_DEVICES="2" python3 run_Limericks_new2.py --search_space 50 --retain_space 5 --prompt "love"
CUDA_VISIBLE_DEVICES="2" python3 run_Limericks_new2.py --search_space 50 --retain_space 5 --prompt "sex"