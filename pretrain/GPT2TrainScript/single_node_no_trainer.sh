#!/bin/bash -l

export HF_HOME=/data/s1/jaehwan/nlp-corpus/.cache/huggingface ### this is the directory path of huggingface cache (default: ~/.cache/huggingface)
DISTRIBUTED_ARGS="--nnodes=1 --nproc_per_node=4 --node-rank=0 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=192.168.0.133:1234" ### node IP
export OMP_NUM_THREADS=16


CUDA_VISIBLE_DEVICES="0,1,2,3" \
MASTER_ADDR=d3 \
torchrun $DISTRIBUTED_ARGS ../run_clm_no_trainer.py \
    --config_name gpt2 \
    --tokenizer_name gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --checkpointing_steps=1000 \
    --output_dir /data/s1/jaehwan/nlp-corpus/gpt2-train-logs \
    --overwrite_cache \
    #--deepspeed ./ds_zero3.json