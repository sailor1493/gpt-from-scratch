#!/bin/bash -l

shard_no=$1
export HF_HOME=/data/s1/chanwoo/hf-home/.cache/huggingface

export OMP_NUM_THREADS=20

DISTRIBUTED_ARGS="--nnodes=2 --nproc_per_node=4 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=d2:1234"

LOCAL_BATCH_SIZE=8
ACCUMULATION_STEP=4
NUM_EPOCH=1
MODEL=skt/kogpt2-base-v2
CKPT_SAVE_ITER=1000
EVAL_SAVE_ITER=4000

mpirun -n 2 \
    -H d2,d3 \
    -bind-to none -map-by slot \
    -x PATH \
    -x OMP_NUM_THREADS \
    -x HF_HOME \
    -mca pml ob1 -mca btl openib \
torchrun $DISTRIBUTED_ARGS pretrain/run_clm.py \
    --config_name $MODEL \
    --tokenizer_name ./tokenizers/mixup_tokenizer \
    --train_file /data/s1/chanwoo/nlp_project/parquet/bulk_books_train.parquet \
    --validation_file /data/s1/chanwoo/nlp_project/parquet/bulk_books_eval.parquet \
    --token False \
    --do_train \
    --do_eval \
    --num_train_epochs $NUM_EPOCH \
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
