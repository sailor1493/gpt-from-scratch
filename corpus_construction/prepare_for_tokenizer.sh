#!/bin/bash

function sample(){
    name=$1
    python sample_corpus.py --src nlp_project/workspace/$name.jsonl --tgt nlp_project/workspace/${name}_sampled_for_tokenizer.jsonl --ratio 0.05
}

# sample bulk_books_shuffled
# sample specialized_shuffled

python merge_same_corpus.py --pttn web-crawl --name web_crawl
python sample_corpus.py --src nlp_project/workspace/web_crawl.jsonl --tgt nlp_project/workspace/web_crawl_for_tokenizer.jsonl --ratio 0.1
