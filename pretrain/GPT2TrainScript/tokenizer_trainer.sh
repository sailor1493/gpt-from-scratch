#!/bin/bash

DATASET=youtube_auto_ko_for_tok

python3 train_tokenizer.py \
    --data-file /data/s1/chanwoo/required/$DATASET.json \
    --tokenizer-name "./${DATASET}_tokenizer" \
