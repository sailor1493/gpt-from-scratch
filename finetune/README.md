# Finetuning

Downstream task using the pretrained model trained with 4 types of data.
1. books
2. specialized (patent, legal)
3. news
4. mixup

The baseline is [skt/kogpt2-base-v2](https://github.com/SKT-AI/KoGPT2)

## Dataset

| Tasks        | Train set | Dev set    | Test set | Metrics  |
|--------------|:---------:|:----------:|:--------:|:--------:|
| nsmc         | 135,000   | 15,000     | 50,000   | accuracy |
| klue_nli     | 22,498    | 2,500      | 3,000    | accuracy |
| klue_ner     | 18,907    | 2,101      | 5,000    | f1       |
| korquad      | 60407     | -          | 5,774    | f1       |

## Run

Use script file
```sh
# For nsmc, klue_nli, klue_ner
./finetune.sh task_name 
```
```bash
# For korquad
./finetune-korquad.sh
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

|               | nsmc (acc) | klue-nli (acc) | klue-ner (f1) | korquad (f1) |
|:-------------:|:----------:|:--------------:|:-------------:|:-------------:
|    baseline   |  0.87352   |     0.6030     |    0.73762    | 0.5173       |
|   bulk_books  |  0.83814   |     0.4267     |    0.63726    | 0.4371       |
|   newspaper   |  0.83874   |     0.4477     |    0.65337    | 0.4720       |
|  specialized  |  0.83626   |     0.4550     |    0.63162    | 0.4622       |
|     mixup     |  0.83580   |     0.4547     |    0.65489    | 0.4627       |

- NSMC: Text classification
- KLUE_NLI: Natural language inference
- KLUE_NER: Named-entity recognition
- KorQuAD: Question Answering
