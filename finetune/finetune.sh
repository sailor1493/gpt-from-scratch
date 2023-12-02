#!/bin/bash -l

MODEL=base
mkdir -p logs/$1/$MODEL
python main.py \
    --task $1 \
    --model_name_or_path skt/kogpt2-base-v2 \
    --num_train_epochs 10 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --output_dir outputs/$1/$MODEL \
    --logging_dir logs/$1/$MODEL > logs/$1/$MODEL/stdout.log 2>&1

for MODEL in "bulk_books" "newspaper" "specialized" "mixup"; do
    mkdir -p logs/$1/$MODEL
    python main.py \
        --task $1 \
        --model_name_or_path /data/s1/chanwoo/nlp_project/logs/$MODEL \
        --num_train_epochs 10 \
        --train_batch_size 16 \
        --eval_batch_size 16 \
        --output_dir outputs/$1/$MODEL \
        --logging_dir logs/$1/$MODEL > logs/$1/$MODEL/stdout.log 2>&1
done
