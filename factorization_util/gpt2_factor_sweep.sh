#!/bin/bash

python3 run_factor.py  \
        --matrix_path gpt2_initial_wte.npy \
        --lr_start "2e-3" \
        --lr_decay "linear" \
        --lr_stop "1e-3" \
        --num_seeds 1 \
        --num_epochs 10000 \
        --A_start 5 \
        --A_step 5 \
        --A_end 100 \
        --output_csv gpt2_factor_sweep.csv

