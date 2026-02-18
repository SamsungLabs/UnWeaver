#!/bin/bash -ex

DATA_DIR=../data/

poetry run python -m data_preprocessing.main --input $DATA_DIR