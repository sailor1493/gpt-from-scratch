#!/bin/bash -l

corpus=bulk_books
function experiment(){
    steps=$1
    bash scripts/additional_training.sh $corpus $steps
}

# experiment 100
experiment 200
experiment 500
experiment 1000