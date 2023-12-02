# Finetuning

Downstream task using the pretrained model trained with 4 types of data.
1. books
2. specialized (patent, legal)
3. news
4. mixup

The baseline is [skt/kogpt2-base-v2](https://github.com/SKT-AI/KoGPT2)

## Dataset

| Tasks            | Train set | Dev set    | Test set | Metrics |
|------------------|:---------:|:----------:|:--------:|:-------:|
| nsmc (19.5MB)    | 135,000     | 15,000        | 50,000    | accuracy      |
| klue_nli     | 22,498     | 2,500        | 3,000    | accuracy      |
| klue_ner     | 18,907     | 2,101        | 5,000    | spearman      |

## Run

Use script file
```sh
./finetune.sh task_name # nsmc, klue_nli
```

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

## Result

### NSMC
Text classification
<!-- TODO -->

### KLUE_NLI
Natural language inference
<!-- TODO -->

### KLUE_NER
Named-entity recognition
<!-- TODO -->