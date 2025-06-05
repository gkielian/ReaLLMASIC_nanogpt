#!/bin/bash
# demos/dataset_colorization.sh

# pushd data/filipino/tagalog_filipino_eng_translation
# bash get_dataset.sh
# popd

# python train.py \
  # --dataset filipino/tagalog_filipino_eng_translation \
  # --compile \
  # --max_iters 10000

python colorize_dataset.py \
  --out_dir        out/inits-True-False-2500-2500-iteration-12-12-768-1024-True-True-1.0-512-True-all-cuda-bfloat16-filipino/tagalog_filipino_eng_translation-True \
  --dataset        filipino/tagalog_filipino_eng_translation  \
  --split          val \
  --num_tokens     2048 \
  --device         cuda:0 \
  --block_size     1024 \
  --mode           minmax  \
  --output_file    kulay_ng_dataset_minmax.txt

python colorize_dataset.py \
  --out_dir        out/inits-True-False-2500-2500-iteration-12-12-768-1024-True-True-1.0-512-True-all-cuda-bfloat16-filipino/tagalog_filipino_eng_translation-True \
  --dataset        filipino/tagalog_filipino_eng_translation  \
  --split          val \
  --num_tokens     2048 \
  --device         cuda:0 \
  --block_size     1024 \
  --mode           softmax  \
  --output_file    kulay_ng_dataset_softmax.txt
