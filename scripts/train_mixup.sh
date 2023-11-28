#!/bin/bash -l

shard_no=$1
export HF_HOME=/data/s1/chanwoo/hf-home/.cache/huggingface

export OMP_NUM_THREADS=20

DISTRIBUTED_ARGS="--nproc_per_node=4 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=localhost:1234"

LOCAL_BATCH_SIZE=8
ACCUMULATION_STEP=8
MODEL=skt/kogpt2-base-v2
NUM_EPOCH=2
CKPT_SAVE_ITER=1000
EVAL_SAVE_ITER=3000
MAX_STEP=10000

torchrun $DISTRIBUTED_ARGS pretrain/run_clm.py \
    --config_name $MODEL \
    --tokenizer_name ./tokenizers/mixup_tokenizer \
    --train_file /data/s1/chanwoo/nlp_project/parquet/mixup_train.parquet \
    --validation_file /data/s1/chanwoo/nlp_project/parquet/mixup_eval.parquet \
    --token False \
    --do_train \
    --do_eval \
    --num_train_epochs $NUM_EPOCH \
    --max_steps $MAX_STEP \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --gradient_accumulation_steps $ACCUMULATION_STEP \
    --output_dir /data/s1/chanwoo/nlp_project/logs/bulk_books \
    --ddp_timeout 18000 \
    --skip_memory_metrics False \
    --save_steps=$CKPT_SAVE_ITER \
    --evaluation_strategy steps \
    --eval_steps=$EVAL_SAVE_ITER \
    --fp16 \
    --preprocessing_num_workers 80
