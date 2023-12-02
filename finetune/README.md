# Finetuning

Downstream task using the pretrained model trained with 4 types of data.
1. books
2. specialized (patent, legal)
3. news
4. mixup

The baseline is [skt/kogpt2-base-v2](https://github.com/SKT-AI/KoGPT2)

## Run

```sh
python main.py \
    --task $TASK \
    --model_name_or_path $MODEL \
    --num_train_epochs $EPOCH \
    --train_batch_size $BATCH_SIZE \
    --eval_batch_size $BATCH_SIZE \
    --output_dir $OUTPUT_DIR \
    --logging_dir $LOG_DIR
```

Or use script file
```sh
./finetune.sh task_name # nsmc, klue_nli
```


## Result

### NSMC
Text classification
<!-- TODO -->

### KLUE_STS
Semantic textual similarity
<!-- TODO -->

### KLUE_NLI
Natural language inference
<!-- TODO -->
