#!/bin/bash
CUDA_VISIBLE_DEVICES="3" python3 run_Limericks_new2.py --search_space 50 --retain_space 5 --prompt "planet"
CUDA_VISIBLE_DEVICES="3" python3 run_Limericks_new2.py --search_space 50 --retain_space 5 --prompt "fire"
CUDA_VISIBLE_DEVICES="3" python3 run_Limericks_new2.py --search_space 50 --retain_space 5 --prompt "animal"
CUDA_VISIBLE_DEVICES="3" python3 run_Limericks_new2.py --search_space 50 --retain_space 5 --prompt "kingdom"
CUDA_VISIBLE_DEVICES="3" python3 run_Limericks_new2.py --search_space 50 --retain_space 5 --prompt "love"
CUDA_VISIBLE_DEVICES="3" python3 run_Limericks_new2.py --search_space 50 --retain_space 5 --prompt "sex"