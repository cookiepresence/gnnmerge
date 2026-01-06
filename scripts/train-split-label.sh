#!/bin/bash

datasets=(
  amazon_computers
  amazon_photo
  cora
  ogbn_arxiv
  # reddit
  wikics
)

models=("gcn" "sage")

for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    echo "Running dataset=${dataset}, model=${model}"

    uv run python3 src/train-split-labels.py \
      --dataset "$dataset" \
      --model "$model" \
      --data-path "artifacts/datasets/${dataset}.pt" \
      --model1-save "artifacts/models/${model}_${dataset}_model_1" \
      --model2-save "artifacts/models/${model}_${dataset}_model_2"
  done
done
