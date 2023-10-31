#!/bin/bash -l

export HF_HOME=/data/s1/jaehwan/nlp-corpus/.cache/huggingface ### this is the directory path of huggingface cache (default: ~/.cache/huggingface)
export OMP_NUM_THREADS=16

DATASET=youtube_ko

python3 train_tokenizer.py \
    --data-file /data/s1/chanwoo/required/$DATASET.json \
    --tokenizer-name "./$DATASET_tokenizer" \
