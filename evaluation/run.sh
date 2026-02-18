#!/bin/bash -ex

datasets=("covidqa" "emanual" "techqa")

for dataset_name in "${datasets[@]}"; do
    INDEX_PATH="../index_${dataset_name}"
    
    poetry run python -m evaluation \
        "$INDEX_PATH" \
        --config configs/custom.json
done
