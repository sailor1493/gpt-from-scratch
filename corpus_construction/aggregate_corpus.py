import random
import os
from tqdm import tqdm

# sample 25% of corpus

dir1 = "/data/s1/chanwoo/release/youtubesubtitles/hyejin/auto_ko"
dir2 = "/data/s1/chanwoo/release/youtubesubtitles/roseanne/auto_ko"

save_path = "nlp_project/workspace/auto_ko_sampled.jsonl"

file_paths = [os.path.join(dir1, file) for file in os.listdir(dir1)] + [
    os.path.join(dir2, file) for file in os.listdir(dir2)
]
with open(save_path, "w") as f1:
    for file in tqdm(file_paths):
        with open(file, "r") as f:
            lines = f.readlines()
            sampled_lines = random.sample(lines, int(len(lines) * 0.25))
        for line in sampled_lines:
            f1.write(line)
