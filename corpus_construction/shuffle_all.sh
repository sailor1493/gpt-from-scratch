#!/bin/bash

function shuffle(){
    name=$1
    python shuffle_corpus.py --src nlp_project/workspace/$name.jsonl --tgt nlp_project/workspace/${name}_shuffled.jsonl
}

shuffle auto_ko_sampled
shuffle bulk_books
shuffle specialized
