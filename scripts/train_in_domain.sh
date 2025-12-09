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

    uv run python3 src/In-domain/train_labelsplits.py \
      --dataset "$dataset" \
      --model "$model" \
      --data-path "artifacts/datasets/${dataset}.pt" \
      --model1-save "artifacts/models/${dataset}_${model}_model1.pt" \
      --model2-save "artifacts/models/${dataset}_${model}_model2.pt" \
      --logs-dir artifacts/logs
  done
done
