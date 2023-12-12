# Pre-training GPT-2 from Scratch

This code is highly based on [Huggingface Transformer](https://github.com/huggingface/transformers) repository for pretraining [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) model.

## Setup
Install in a Conda env with `PyTorch` and `CUDA` available.

```bash
pip install -r requirements.txt
```

## Run
This code runs pr-training GPT-2 (125M) from scratch in DDP mode. 

You have to newly set up the environment variables in the script (e.g., `HF_HOME`, `DISTRIBUTED_ARGS`, `MASTER_ADDR`) depending on your system otherwise PyTorch DDP will not work.

### Single Node Training
```bash
# With Trainer
bash single_node_trainer.sh
```
```bash
# Without Trainer
bash single_node_no_trainer.sh
```

### Multi Node Training
```bash
# On the master node 
bash multi_node_trainer.sh
```


