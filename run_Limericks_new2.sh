#!/bin/bash
CUDA_VISIBLE_DEVICES="3" python3 run_Limericks_new2.py --search_space 250 --retain_space 3 --prompt "planet"
CUDA_VISIBLE_DEVICES="3" python3 run_Limericks_new2.py --search_space 250 --retain_space 3 --prompt "fire"
CUDA_VISIBLE_DEVICES="3" python3 run_Limericks_new2.py --search_space 250 --retain_space 3 --prompt "animal"
CUDA_VISIBLE_DEVICES="3" python3 run_Limericks_new2.py --search_space 250 --retain_space 3 --prompt "kingdom"
CUDA_VISIBLE_DEVICES="3" python3 run_Limericks_new2.py --search_space 250 --retain_space 3 --prompt "love"
CUDA_VISIBLE_DEVICES="3" python3 run_Limericks_new2.py --search_space 250 --retain_space 3 --prompt "sex"