#/bin/bash

# head to repo root
cd ../

softmax_variation=("softermax" "polymax" "constantmax")

# do one iteration with regular softmax
softmax_variant="regular_softmax"
python3 train.py \
  --max_iters 16000 \
  --eval_iters 200 \
  --eval_interval 200 \
  --log_interval 10 \
  --dataset "tinystories_en" \
  --use_post_ln \
  --no-use_softmax_variant \
  --tensorboard_project "ts_en_${softmax_variant}_explorations_post_ln" \
  --tensorboard_run_name "${softmax_variant}" \
  --block_size 1024 \
  --out_dir "ts_en_${softmax_variant}_evaluations_post_ln" \
  --compile

# Loop over the array
for softmax_variant in "${softmax_variation[@]}"
do
  softmax_variant="$element"

  python3 train.py \
    --max_iters 16000 \
    --eval_iters 200 \
    --eval_interval 200 \
    --log_interval 10 \
    --dataset "tinystories_en" \
    --use_post_ln \
    --no-use_softmax_variant \
    --tensorboard_project "ts_en_${softmax_variant}_explorations_post_ln" \
    --tensorboard_run_name "${softmax_variant}" \
    --block_size 1024 \
    --out_dir "ts_en_${softmax_variant}_evaluations_post_ln" \
    --compile
done
