# Finetuning

Downstream task using the pretrained model trained with 4 types of data.
1. books
2. specialized (patent, legal)
3. news
4. mixup

The baseline is [skt/kogpt2-base-v2](https://github.com/SKT-AI/KoGPT2)

## NSMC

Text classification

```sh
python nsmc.py \
    --model_name_or_path ${MODEL_PATH} \
    --num_train_epochs 10
```

## KLUE-STS

Semantic textual similarity
<!-- TODO -->

## KLUE-NLI

Natural language inference
<!-- TODO -->
