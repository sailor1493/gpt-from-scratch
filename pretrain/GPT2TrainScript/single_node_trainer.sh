#!/bin/bash -l

export HF_HOME=/data/s1/jaehwan/nlp-corpus/.cache/huggingface ### this is the directory path of huggingface cache (default: ~/.cache/huggingface)
DISTRIBUTED_ARGS="--nnodes=1 --nproc_per_node=4 --node-rank=0 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=192.168.0.132:1234" # node IP
export OMP_NUM_THREADS=16


CUDA_VISIBLE_DEVICES="0,1,2,3" \
MASTER_ADDR=d2 \
torchrun $DISTRIBUTED_ARGS ../run_clm.py \
    --config_name gpt2 \
    --tokenizer_name gpt2 \
    --train_file /data/s1/chanwoo/required/youtube_ko.json \
    --token False \
    --do_train \
    --do_eval \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --save_steps=1000 \
    --output_dir /data/s1/jaehwan/nlp-corpus/gpt2-train-logs \
    --overwrite_output_dir \
    --ddp_timeout 18000 ### default: 1800

    #--model_name_or_path gpt2 \
    #--deepspeed ./ds_zero3.json
    # --dataset_name wikitext \
    # --dataset_config_name wikitext-2-raw-v1 \
