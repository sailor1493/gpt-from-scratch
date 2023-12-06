#!/bin/bash

function sample(){
    name=$1
    python sample_corpus.py --src nlp_project/experiment_corpus/${name}_final.json --tgt nlp_project/experiment_corpus/${name}_additional.json --ratio 0.2
}

sample bulk_books
sample specialized
sample mixup
sample newspaper
