#!/bin/bash -l


# export HF_HOME=/data/s1/jaehwan/nlp-corpus/.cache/huggingface

export OMP_NUM_THREADS=20

DISTRIBUTED_ARGS="--nnodes=2 --nproc_per_node=4 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=d2:1234"

DATASET=youtube_auto_ko
LOCAL_BATCH_SIZE=32
NUM_EPOCH=1
MODEL=gpt2
CKPT_SAVE_ITER=1000
EVAL_SAVE_ITER=1000

mpirun -n 2 \
    -H d2,d3 \
    -bind-to none -map-by slot \
    -x PATH \
    -x OMP_NUM_THREADS \
    -mca pml ob1 -mca btl openib \
torchrun $DISTRIBUTED_ARGS run_clm.py \
    --config_name $MODEL \
    --tokenizer_name ../tokenizers/youtube_auto_ko \
    --train_file /data/s1/chanwoo/nlp_project/parquet/ko_train.parquet \
    --validation_file /data/s1/chanwoo/nlp_project/parquet/ko_eval.parquet \
    --token False \
    --do_train \
    --do_eval \
    --num_train_epochs $NUM_EPOCH \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --output_dir /data/s1/chanwoo/nlp_project/logs/multinode \
    --overwrite_output_dir \
    --ddp_timeout 18000 \
    --skip_memory_metrics False \
    --save_steps=$CKPT_SAVE_ITER \
    --evaluation_strategy steps \
    --eval_steps=$EVAL_SAVE_ITER \
    --gradient_checkpointing \
    --fp16 \
    --preprocessing_num_workers 80
