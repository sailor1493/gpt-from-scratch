#!/bin/bash -l

source ~/anaconda3/etc/profile.d/conda.sh
conda activate gpt2
export HF_HOME=/data/s1/jaehwan/nlp-corpus/.cache/huggingface ### this is the directory path of huggingface cache (default: ~/.cache/huggingface)

DISTRIBUTED_ARGS="--nnodes=2 --nproc_per_node=4 --node-rank=1 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=192.168.0.133:1234"
BATCH_SIZE=4


OMP_NUM_THREADS=16 \
NCCL_IB_GID_INDEX=3 \
torchrun $DISTRIBUTED_ARGS ../run_clm.py \
    --config_name gpt2 \
    --tokenizer_name gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --save_steps=1000 \
    --output_dir /data/s1/jaehwan/nlp-corpus/gpt2-train-logs \
    --overwrite_output_dir \
    #--gradient_checkpointing \
    #--deepspeed ./ds_zero3.json