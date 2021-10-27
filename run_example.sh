#!/bin/bash

echo "Start Training"

python main.py \
  --test False \
  --max_len 50 \
  --batch_size 256 \
  --epochs 3 \
  --eval_steps 250 \
  --lr 0.0001 \
  --warmup_ratio 0.05 \
  --temperature 0.05 \
  --path_to_data ./data/ \
  --train_data train_nli_sample \
  --valid_data valid_sts_sample

echo "Start Testing"

python main.py \
  --train False \
  --test True \
  --max_len 50 \
  --batch_size 256 \
  --epochs 3 \
  --eval_steps 250 \
  --lr 0.00005 \
  --warmup_ratio 0.05 \
  --temperature 0.05 \
  --path_to_data ./data/ \
  --test_data test_sts_sample \
  --path_to_saved_model output/best_checkpoint.pt

echo "Semantic Search"

python SemanticSearch.py

