#!/bin/bash -l

export LM_EVAL_PATH=~/repo/lm-evaluation-harness/
export TOKENIZERS_PARALLELISM=false

MODEL=/data/s1/chanwoo/nlp_project/logs/$1
GPU_NO=$2

RESULT_DIR=logs/kobest_$1
NUM_FEWSHOT="0 5 10"

echo "mkdir -p $RESULT_DIR"
mkdir -p $RESULT_DIR

for i in $NUM_FEWSHOT; do
    python -W ignore $LM_EVAL_PATH/main.py \
        --model gpt2 \
        --model_args pretrained=$MODEL \
        --tasks kobest_boolq,kobest_copa,kobest_hellaswag,kobest_sentineg,kobest_wic \
        --num_fewshot $i \
        --device cuda:$GPU_NO \
        --no_cache \
        --output_path $RESULT_DIR/$i-shot.json > $RESULT_DIR/$i-shot.log 2>&1
done
