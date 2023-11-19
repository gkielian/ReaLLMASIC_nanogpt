#/bin/bash

# head to repo root
cd ../

# start training
# python3 train.py \
#   --max_iters 8000 \
#   --eval_iters 200 \
#   --eval_interval 100 \
#   --log_interval 10 \
#   --dataset "tinystories_en" \
#   --use_post_ln \
#   --use_softmax_variant \
#   --softmax_variant "constantmax" \
#   --tensorboard_project "ts_en_softmax_explorations_post_ln" \
#   --tensorboard_run_name "consmax_base_e" \
#   --block_size 2048 \
#   --out_dir "ts_en_consmax_evaluations_post_ln" \
#   --compile

# # start training
# python3 train.py \
#   --max_iters 8000 \
#   --eval_iters 200 \
#   --eval_interval 100 \
#   --log_interval 10 \
#   --dataset "tinystories_en" \
#   --use_post_ln \
#   --use_softmax_variant \
#   --softmax_variant "polymax" \
#   --tensorboard_project "ts_en_polymax_explorations_post_ln" \
#   --tensorboard_run_name "polymax" \
#   --block_size 1024 \
#   --out_dir "ts_en_polymax_evaluations_post_ln" \
#   --compile

softmax_variant="regular_softermax"
  # --use_softermax_xmax \
  # --softmax_variant "${softmax_variant}" \

python3 train.py \
  --max_iters 8000 \
  --eval_iters 200 \
  --eval_interval 100 \
  --log_interval 10 \
  --dataset "tinystories_en" \
  --use_post_ln \
  --no-use_softmax_variant \
  --tensorboard_project "ts_en_${softmax_variant}_explorations_post_ln" \
  --tensorboard_run_name "${softmax_variant}" \
  --block_size 1024 \
  --out_dir "ts_en_${softmax_variant}_evaluations_post_ln" \
  --compile

softmax_variant="polymax"
  # --use_softermax_xmax \
  # --softmax_variant "${softmax_variant}" \

python3 train.py \
  --max_iters 8000 \
  --eval_iters 200 \
  --eval_interval 100 \
  --log_interval 10 \
  --dataset "tinystories_en" \
  --use_post_ln \
  --softmax_variant "${softmax_variant}" \
  --tensorboard_project "ts_en_${softmax_variant}_explorations_post_ln" \
  --tensorboard_run_name "${softmax_variant}" \
  --block_size 1024 \
  --out_dir "ts_en_${softmax_variant}_evaluations_post_ln" \
  --compile

softmax_variant="constantmax"
  # --use_softermax_xmax \
  # --softmax_variant "${softmax_variant}" \

python3 train.py \
  --max_iters 8000 \
  --eval_iters 200 \
  --eval_interval 100 \
  --log_interval 10 \
  --dataset "tinystories_en" \
  --use_post_ln \
  --softmax_variant "${softmax_variant}" \
  --use_softermax_xmax \
  --tensorboard_project "ts_en_${softmax_variant}_explorations_post_ln" \
  --tensorboard_run_name "${softmax_variant}" \
  --block_size 1024 \
  --out_dir "ts_en_${softmax_variant}_evaluations_post_ln" \
  --compile

softmax_variant="softermax"
  # --use_softermax_xmax \
  # --softmax_variant "${softmax_variant}" \

python3 train.py \
  --max_iters 8000 \
  --eval_iters 200 \
  --eval_interval 100 \
  --log_interval 10 \
  --dataset "tinystories_en" \
  --use_post_ln \
  --softmax_variant "${softmax_variant}" \
  --use_softermax_xmax \
  --tensorboard_project "ts_en_${softmax_variant}_explorations_post_ln" \
  --tensorboard_run_name "${softmax_variant}" \
  --block_size 1024 \
  --out_dir "ts_en_${softmax_variant}_evaluations_post_ln" \
  --compile

softmax_variant="softermax"
  # --use_softermax_xmax \
  # --softmax_variant "${softmax_variant}" \

python3 train.py \
  --max_iters 8000 \
  --eval_iters 200 \
  --eval_interval 100 \
  --log_interval 10 \
  --dataset "tinystories_en" \
  --use_post_ln \
  --softmax_variant "${softmax_variant}" \
  --no-use_softermax_xmax \
  --tensorboard_project "ts_en_${softmax_variant}_explorations_post_ln_no_xmax" \
  --tensorboard_run_name "${softmax_variant}" \
  --block_size 1024 \
  --out_dir "ts_en_${softmax_variant}_evaluations_post_ln_no_xmax" \
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
