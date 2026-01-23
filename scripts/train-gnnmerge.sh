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

classes=(0 1)

hidden_dims=(8 32 128 512)

for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    for hidden_dim in "${hidden_dims[@]}"; do

      model0_path="artifacts/models/${model}_${dataset}_model_0-2_${hidden_dim}dims"
      model1_path="artifacts/models/${model}_${dataset}_model_1-2_${hidden_dim}dims"
      save_path="artifacts/merged/${model}_${dataset}_nc_${hidden_dim}dims"

      echo "Merging:"
      echo "  ${model0_path}"
      echo "  ${model1_path}"
      echo "  -> ${save_path}"

      uv run python3 src/gnnmerge.py \
        --model-path "${model0_path}" \
        --model-path "${model1_path}" \
        --save-path "${save_path}"
      echo "done!"
    done
  done
done
