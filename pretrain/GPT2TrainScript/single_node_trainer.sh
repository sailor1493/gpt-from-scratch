#!/bin/bash -l

# export HF_HOME=/data/s1/jaehwan/nlp-corpus/.cache/huggingface ### this is the directory path of huggingface cache (default: ~/.cache/huggingface)
DISTRIBUTED_ARGS="--nnodes=1 --nproc_per_node=4 --node-rank=0 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=localhost:1234" # node IP
export OMP_NUM_THREADS=16

### Tokenization
# DATASET=youtube_ko

# python3 train_tokenizer.py \
#     --data-file /data/s1/chanwoo/required/$DATASET.json \
#     --tokenizer-name "./$DATASET_tokenizer"

### Pretraining
DATASET=youtube_auto_ko
LOCAL_BATCH_SIZE=32
NUM_EPOCH=1
MODEL=gpt2
CKPT_SAVE_ITER=500
EVAL_SAVE_ITER=500

CUDA_VISIBLE_DEVICES="0,1,2,3" \
torchrun $DISTRIBUTED_ARGS ../run_clm.py \
    --config_name $MODEL \
    --tokenizer_name ../../tokenizers/bml-test \
    --train_file /data/s1/chanwoo/required/$DATASET.json \
    --token False \
    --do_train \
    --do_eval \
    --num_train_epochs $NUM_EPOCH \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --output_dir /data/s1/chanwoo/nlp_project/logs/singlenode \
    --overwrite_output_dir \
    --ddp_timeout 18000 \
    --skip_memory_metrics False \
    --save_steps=$CKPT_SAVE_ITER \
    --evaluation_strategy steps \
    --eval_steps=$EVAL_SAVE_ITER \
    --gradient_checkpointing \
    --fp16
