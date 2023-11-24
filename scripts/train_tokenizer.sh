#!/bin/bash

python pretrain/train_tokenizer.py \
    --data-file corpus_construction/nlp_project/experiment_corpus/tokenizer_corpus.jsonl \
    --tokenizer-name ./tokenizers/mixup_tokenizer