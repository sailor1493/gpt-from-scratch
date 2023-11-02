import random

src_corpus = "/data/s1/chanwoo/required/youtube_auto_ko.json"
dest_corpus = "/data/s1/chanwoo/required/youtube_auto_ko_for_tok.json"
RATIO = 0.05
count = 0
with open(src_corpus, "r") as f1, open(dest_corpus, "w") as f2:
    for line in f1:
        # only 1/10 of the corpus
        count += 1
        if count % 10000 == 0:
            print(count)
        if random.random() > RATIO:
            continue
        f2.write(line)
