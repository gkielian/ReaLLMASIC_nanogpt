#/bin/bash

# head to repo root
cd ../

# start training
python3 train.py \
  --max_iters 8000 \
  --eval_iters 200 \
  --eval_interval 100 \
  --log_interval 10 \
  --n_layer 24 \
  --n_head 16 \
  --n_embd 1024 \
  --dataset "openwebtext" \
  --use_post_ln \
  --use_softmax_variant \
  --softmax_variant "constantmax" \
  --tensorboard_project "shakes_tiktoken_temperature" \
  --tensorboard_run_name "constmax_temperature" \
  --block_size 2048 \
  --out_dir "constantmax_openwebtext_temp" \
  --compile

# start training
# python3 train.py \
#   --max_iters 8000 \
#   --eval_iters 200 \
#   --eval_interval 100 \
#   --log_interval 10 \
#   --dataset "tinystories_en" \
#   --use_post_ln \
#   --use_softmax_variant \
#   --use_softermax_xmax \
#   --softmax_variant "softermax" \
#   --tensorboard_project "ts_en_softmax_explorations_post_ln_xmax" \
#   --tensorboard_run_name "softermax" \
#   --block_size 2048 \
#   --out_dir "ts_en_softermax_evaluations_post_ln" \
#   --compile

# # start training
# python3 train.py \
#   --max_iters 8000 \
#   --eval_iters 200 \
#   --eval_interval 100 \
#   --log_interval 10 \
#   --dataset "tinystories_en" \
#   --use_post_ln \
#   --no-use_softmax_variant \
#   --tensorboard_project "ts_en_softmax_explorations_post_ln" \
#   --tensorboard_run_name "softmax" \
#   --block_size 2048 \
#   --out_dir "ts_en_softmax_evaluations_post_ln" \
#   --compile
