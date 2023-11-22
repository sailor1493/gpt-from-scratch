#!/bin/bash

function pipeline(){
    name=$1
    tokenizer_name=./tokenizers/$name
    data_path=corpus_construction/nlp_project/workspace/${name}_shuffled.jsonl
    tokenizer_data_path=corpus_construction/nlp_project/workspace/${name}_shuffled_sampled.jsonl
    log_path=logs/$name.log
    # python pretrain/train_tokenizer.py \
    #     --data-file $tokenizer_data_path \
    #     --tokenizer-name $tokenizer_name
    python corpus_construction/token_count.py \
        --tokenizer-name $tokenizer_name \
        --data-file $tokenizer_data_path \
        --log-file $log_path
}


pipeline bulk_books
pipeline specialized

exit 0


name=bulk_books
tokenizer_name=./tokenizers/$name
data_path=corpus_construction/nlp_project/workspace/${name}_shuffled.jsonl
tokenizer_data_path=corpus_construction/nlp_project/workspace/${name}_shuffled_sampled.jsonl
log_path=logs/$name.log

python corpus_construction/token_count.py \
    --tokenizer-name $tokenizer_name \
    --data-file $tokenizer_data_path \
    --log-file $log_path
