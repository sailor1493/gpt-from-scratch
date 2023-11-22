#!/bin/bash

function sample(){
    name=$1
    python sample_corpus.py --src nlp_project/workspace/$name.jsonl --tgt nlp_project/workspace/${name}_sampled.jsonl
}

sample bulk_books_shuffled
sample specialized_shuffled
