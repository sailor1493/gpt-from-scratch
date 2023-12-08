import random

original_corpus = "nlp_project/workspace/auto_ko_sampled.jsonl"
sampled_corpus = "nlp_project/workspace/auto_ko_for_tokenizer.jsonl"

with open(original_corpus, "r") as f1, open(sampled_corpus, "w") as f2:
    for line in f1:
        if random.random() < 0.1:
            f2.write(line)
