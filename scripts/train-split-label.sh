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
	for chosen_class in "${classes[@]}"; do
	    for hidden_dim in "${hidden_dims[@]}"; do
		echo "Running dataset=${dataset}, model=${model}"
		
		uv run python3 src/train_node_classification.py \
		   --dataset "$dataset" \
		   --model "$model" \
		   --data-path "artifacts/datasets/${dataset}.pt" \
		   --num-classes 2 \
		   --chosen-class "${chosen_class}" \
		   --hidden-dim "${hidden_dim}" \
		   --num-epochs 150 \
		   --save-path "artifacts/models/${model}_${dataset}_model_${chosen_class}-2_${hidden_dim}dims" \
		   --wandb-name "${model}_${dataset}_model_${chosen_class}-2_${hidden_dim}dims"
	    done
	done
    done
done
