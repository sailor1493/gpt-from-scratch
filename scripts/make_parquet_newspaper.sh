#!/bin/bash -l

export HF_HOME=/data/s1/chanwoo/hf-home/.cache/huggingface

### Preprocessing
LOCAL_BATCH_SIZE=32
NUM_EPOCH=1
MODEL=skt/kogpt2-base-v2
CKPT_SAVE_ITER=200
EVAL_SAVE_ITER=200

CUDA_VISIBLE_DEVICES="0,1,2,3" \
python pretrain/preprocess_only.py \
    --config_name $MODEL \
    --tokenizer_name ./tokenizers/mixup_tokenizer \
    --train_file /data/s1/chanwoo/nlp_project/experiment_corpus/newspaper.json \
    --token False \
    --do_train \
    --do_eval \
    --num_train_epochs $NUM_EPOCH \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --output_dir /data/s1/chanwoo/nlp_project/logs/singlenode \
    --overwrite_output_dir \
    --skip_memory_metrics False \
    --save_steps $CKPT_SAVE_ITER \
    --evaluation_strategy steps \
    --eval_steps $EVAL_SAVE_ITER \
    --gradient_checkpointing \
    --fp16 \
    --preprocessing_num_workers 80 \
    --overwrite_cache True \
    --validation_split_percentage 5
