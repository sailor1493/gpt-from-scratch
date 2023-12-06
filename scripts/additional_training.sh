#!/bin/bash -l

export HF_HOME=/data/s1/chanwoo/hf-home/.cache/huggingface
export OMP_NUM_THREADS=20
DISTRIBUTED_ARGS="--nproc_per_node=4 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=localhost:1234"


additional_corpus_name=$1
steps=$2
base_corpus=$3

if [ -z "$base_corpus" ]
then
    echo "base_corpus is empty. Setting to mixup"
    base_corpus="mixup"
fi


model_dir=/data/s1/chanwoo/nlp_project/logs
model=$model_dir/$base_corpus
corpus_dir=/data/s1/chanwoo/nlp_project/experiment_corpus
additional_corpus="$corpus_dir/${additional_corpus_name}_additional.json"
experiment_name="${base_corpus}_${additional_corpus_name}_${steps}"


LOCAL_BATCH_SIZE=4
ACCUMULATION_STEP=16
MODEL=$model
NUM_EPOCH=20
CKPT_SAVE_ITER=100
EVAL_SAVE_ITER=100
MAX_STEP=$steps

torchrun $DISTRIBUTED_ARGS pretrain/additional_training.py \
    --model_name_or_path $MODEL \
    --tokenizer_name ./tokenizers/mixup_tokenizer \
    --train_file $additional_corpus \
    --token False \
    --do_train \
    --do_eval \
    --validation_split_percentage 1 \
    --num_train_epochs $NUM_EPOCH \
    --max_steps $MAX_STEP \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --gradient_accumulation_steps $ACCUMULATION_STEP \
    --output_dir /data/s1/chanwoo/nlp_project/logs/$experiment_name \
    --ddp_timeout 18000 \
    --skip_memory_metrics False \
    --save_steps=$CKPT_SAVE_ITER \
    --evaluation_strategy steps \
    --eval_steps=$EVAL_SAVE_ITER \
    --fp16 \
    --preprocessing_num_workers 64 \
