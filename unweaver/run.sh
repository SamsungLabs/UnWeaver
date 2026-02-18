#!/bin/bash -ex

export MONGO_USER="MONGO_USER"
export MONGO_PASSWORD="MONGO_PASSWORD"
export MONGO_URL="MONGO_URL:PORT"

datasets=("covidqa" "emanual" "techqa")

for dataset_name in "${datasets[@]}"; do
    QUESTIONS_PATH="../data/${dataset_name}/questions.json"
    DATA_PATH="../data/${dataset_name}/files_preprocessed/"
    INDEX_PATH="../index_${dataset_name}"
    
    poetry run python -m unweaver.index \
        "$DATA_PATH" \
        "$INDEX_PATH" \
        --config configs/custom.json
    
    poetry run python -m unweaver.query \
        "$QUESTIONS_PATH" \
        "$INDEX_PATH" \
        --run_name unweaver_run \
        --config configs/custom.json
done
