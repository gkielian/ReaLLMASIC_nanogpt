#!/bin/bash


# Learned Dataset Embedding
## very small footprint, only an addition
## applies best to layer 0
python3 train.py \
  --dataset xsum \
  --dataset_list ko_commercial fr_wikipedia wikipedia_indonesian xsum \
  --dataset_sampling_probs 1 1 1 1 \
  --use_lsv \
  --max_sample_tokens 100 \
  --max_iters 2500 \
  --sample_each_eval \
  --apply_lsv_at_layer_idx 0 \
  --eval_interval 500 \
  --eval_iters 50 \
  --dataset_interleaving \
  --dataset_interleaving_shuffle \
  --lsv_variant one_hot \
  --out_dir "out_one_hot" \
  --init_from "scratch"

# Learned Steering Vectors
## Relatively small footprint, small mlp (FIRE inspired)
## Tested on multiple layers, can work at layer 5
## Finetunes very quickly (500 iterations finetuning for 124M GPT2)
python3 train.py \
  --dataset xsum \
  --dataset_list ko_commercial fr_wikipedia wikipedia_indonesian xsum \
  --dataset_sampling_probs 1 1 1 1 \
  --use_lsv \
  --block_size 256 \
  --batch_size 8 \
  --learning_rate "6e-4" \
  --dropout "0.1" \
  --max_sample_tokens 100 \
  --max_iters 2500 \
  --sample_each_eval \
  --apply_lsv_at_layer_idx 6 \
  --eval_interval 500 \
  --eval_iters 100 \
  --dataset_interleaving \
  --dataset_interleaving_shuffle \
  --lsv_variant "one_hot_mlp" \
  --out_dir "out_one_hot_mlp" \
  --tensorboard_run_name "out_one_hot_mlp" \
  --init_from "gpt2*"
