#!/bin/bash -l

DISTRIBUTED_ARGS="--nnodes=2 --nproc_per_node=1 --rdzv_id=42 --rdzv_backend=c10d --rdzv_endpoint=d2"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate nlp-corpus

OMP_NUM_THREADS=16
torchrun $DISTRIBUTED_ARGS TestPyTorchMPI.py