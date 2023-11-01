#!/bin/bash -l


# export HF_HOME=/data/s1/jaehwan/nlp-corpus/.cache/huggingface

DISTRIBUTED_ARGS="--nnodes=2 --nproc_per_node=4 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=d2:1234"
BATCH_SIZE=4


OMP_NUM_THREADS=16 \

torchrun $DISTRIBUTED_ARGS ../run_clm.py \
    --config_name $MODEL \
    --tokenizer_name ../../tokenizers/bml-test \
    --train_file /data/s1/chanwoo/required/$DATASET.json \
    --token False \
    --do_train \
    --do_eval \
    --num_train_epochs $NUM_EPOCH \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --output_dir logs \
    --overwrite_output_dir \
    --ddp_timeout 18000 \
    --skip_memory_metrics False \
    --save_steps=$CKPT_SAVE_ITER \
    --evaluation_strategy steps \
    --eval_steps=$EVAL_SAVE_ITER \
    --gradient_checkpointing \
    --fp16
