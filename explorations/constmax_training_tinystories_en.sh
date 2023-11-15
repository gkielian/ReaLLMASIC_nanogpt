#/bin/bash

# head to repo root
cd ../

# start training
python3 train.py \
  --max_iters 8000 \
  --eval_iters 200 \
  --eval_interval 100 \
  --log_interval 10 \
  --dataset "tinystories_en" \
  --use_softmax_variant \
  --softmax_variant "constantmax" \
  --tensorboard_project "ts_en_softmax_explorations" \
  --tensorboard_run_name "consmax_base_e" \
  --block_size 2048 \
  --out_dir "ts_en_consmax_evaluations" \
  --compile

# start training
python3 train.py \
  --max_iters 8000 \
  --eval_iters 200 \
  --eval_interval 100 \
  --log_interval 10 \
  --dataset "tinystories_en" \
  --use_softmax_variant \
  --softmax_variant "softermax" \
  --tensorboard_project "ts_en_softmax_explorations" \
  --tensorboard_run_name "softermax" \
  --block_size 2048 \
  --out_dir "ts_en_softermax_evaluations" \
  --compile

# start training
python3 train.py \
  --max_iters 8000 \
  --eval_iters 200 \
  --eval_interval 100 \
  --log_interval 10 \
  --dataset "tinystories_en" \
  --no-use_softmax_variant \
  --tensorboard_project "ts_en_softmax_explorations" \
  --tensorboard_run_name "softmax" \
  --block_size 2048 \
  --out_dir "ts_en_softmax_evaluations" \
  --compile
