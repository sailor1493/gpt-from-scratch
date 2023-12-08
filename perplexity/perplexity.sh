#!/bin/bash -l

export HF_HOME=/data/s1/jaehwan/nlp-corpus/.cache/huggingface


baseline="
        skt/kogpt2-base-v2
        /data/s1/chanwoo/nlp_project/logs/bulk_books
        /data/s1/chanwoo/nlp_project/logs/specialized
        /data/s1/chanwoo/nlp_project/logs/newspaper
        /data/s1/chanwoo/nlp_project/logs/mixup
        /data/s1/chanwoo/nlp_project/logs/bulk_books_additional_100
        /data/s1/chanwoo/nlp_project/logs/bulk_books_additional_200
        /data/s1/chanwoo/nlp_project/logs/bulk_books_additional_500
        /data/s1/chanwoo/nlp_project/logs/bulk_books_additional_1000
        /data/s1/chanwoo/nlp_project/logs/specialized_additional_100
        /data/s1/chanwoo/nlp_project/logs/specialized_additional_200
        /data/s1/chanwoo/nlp_project/logs/specialized_additional_500
        /data/s1/chanwoo/nlp_project/logs/specialized_additional_1000
        /data/s1/chanwoo/nlp_project/logs/newspaper_additional_100
        /data/s1/chanwoo/nlp_project/logs/newspaper_additional_200
        /data/s1/chanwoo/nlp_project/logs/newspaper_additional_500
        /data/s1/chanwoo/nlp_project/logs/newspaper_additional_1000
        /data/s1/chanwoo/nlp_project/logs/mixup_additional_100
        /data/s1/chanwoo/nlp_project/logs/mixup_additional_200
        /data/s1/chanwoo/nlp_project/logs/mixup_additional_500
        /data/s1/chanwoo/nlp_project/logs/mixup_additional_1000
        "

for model in $baseline; do
    echo python kogpt2_perplexity.py $model
    python kogpt2_perplexity.py $model
done