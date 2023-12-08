#!/bin/bash

function tokenizer_train(){
    name=$1
    python train_tokenizer.py \
        --data-file ../corpus_construction/nlp_project/workspace/${name}_shuffled_sampled.jsonl \
        --tokenizer-name $name
}

tokenizer_train bulk_books
tokenizer_train specialized