#/bin/bash

# head to repo root
cd ../

for seed in {1337..1387..2}; do

  python3 train.py \
    --max_iters 8000 \
    --eval_iters 200 \
    --eval_interval 100 \
    --log_interval 10 \
    --n_layer 24 \
    --n_head 16 \
    --n_embd 1024 \
    --seed_offset "$seed" \
    --dataset "openwebtext" \
    --use_post_ln \
    --use_softmax_variant \
    --softmax_variant "constantmax" \
    --tensorboard_project "owt_tiktoken_temperature" \
    --tensorboard_run_name "constmax_temperature_${seed}" \
    --block_size 2048 \
    --out_dir "constantmax_openwebtext_temp" \
    --compile

done
