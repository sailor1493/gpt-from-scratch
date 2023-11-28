#!/bin/bash

# python corpus_construction/sample_corpus.py \
#     --src corpus_construction/nlp_project/experiment_corpus/bulk_books_shuffled.json \
#     --tgt corpus_construction/nlp_project/experiment_corpus/bulk_books_for_mixup.json \
#     --ratio 0.25
python corpus_construction/sample_corpus.py \
    --src corpus_construction/nlp_project/experiment_corpus/specialized_shuffled.json \
    --tgt corpus_construction/nlp_project/experiment_corpus/specialized_for_mixup.json \
    --ratio 0.25
# python corpus_construction/sample_corpus.py \
#     --src corpus_construction/nlp_project/experiment_corpus/newspaper.json \
#     --tgt corpus_construction/nlp_project/experiment_corpus/newspaper_for_mixup.json \
#     --ratio 0.5

# touch corpus_construction/nlp_project/experiment_corpus/mixup_corpus.json
# cat corpus_construction/nlp_project/experiment_corpus/bulk_books_for_mixup.json >> corpus_construction/nlp_project/experiment_corpus/mixup_corpus.json
cat corpus_construction/nlp_project/experiment_corpus/specialized_for_mixup.json >> corpus_construction/nlp_project/experiment_corpus/mixup_corpus_shuffled.json
# cat corpus_construction/nlp_project/experiment_corpus/newspaper_for_mixup.json >> corpus_construction/nlp_project/experiment_corpus/mixup_corpus.json

python corpus_construction/shuffle_corpus.py \
    --src corpus_construction/nlp_project/experiment_corpus/mixup_corpus_shuffled.json \
    --tgt corpus_construction/nlp_project/experiment_corpus/mixup_corpus_shuffled_2.json

# remove intermediate files
# rm corpus_construction/nlp_project/experiment_corpus/bulk_books_for_mixup.json
# rm corpus_construction/nlp_project/experiment_corpus/specialized_for_mixup.json
# rm corpus_construction/nlp_project/experiment_corpus/newspaper_for_mixup.json
rm corpus_construction/nlp_project/experiment_corpus/mixup_corpus_shuffled.json

mv corpus_construction/nlp_project/experiment_corpus/mixup_corpus_shuffled_2.json corpus_construction/nlp_project/experiment_corpus/mixup_corpus_shuffled.json