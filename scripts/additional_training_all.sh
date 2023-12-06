#!/bin/bash -l

function experiment(){
    corpus=$1
    steps=$2
    bash scripts/additional_training.sh $corpus $steps
}

corpuses=(bulk_books specialized newspaper mixup)
# corpuses=(bulk_books)
for corpus in "${corpuses[@]}"
do
    experiment $corpus 100
    experiment $corpus 200
    experiment $corpus 500
    experiment $corpus 1000
done