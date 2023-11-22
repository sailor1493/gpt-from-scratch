#!/bin/bash

function shuffle(){
    name=$1
    python shuffle_corpus.py --src nlp_project/workspace/$name.jsonl --tgt nlp_project/workspace/${name}_shuffled.jsonl
}

shuffle bulk_books
shuffle specialized
shuffle auto_ko_sampled
