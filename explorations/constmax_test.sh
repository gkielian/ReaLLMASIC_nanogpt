#/bin/bash

# head to repo root
cd ../

pushd ../data/shakespeare/

python3 prepare.py

popd

for seed in {1337..1347..2}; do
  # TODO: find larger model that aggrees with A100, maybe try:
  # --n_layer 12 \
  # --n_head 12 \
  # --n_embd 768 \

  python3 train.py \
    --max_iters 8000 \
    --eval_iters 200 \
    --eval_interval 100 \
    --log_interval 10 \
    --seed_offset "$seed" \
    --dataset "shakespeare" \
    --use_post_ln \
    --use_softmax_variant \
    --softmax_variant "constantmax" \
    --tensorboard_project "shakespeare_tiktoken" \
    --tensorboard_run_name "constmax_temperature_${seed}" \
    --block_size 1024 \
    --out_dir "constantmax_ts_temp" \
    --compile

done
