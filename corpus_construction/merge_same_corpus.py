import os
from argparse import ArgumentParser
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("--pttn", type=str, required=True)
parser.add_argument("--name", type=str, required=True)
args = parser.parse_args()

base_dir = "nlp_project/corpus/yna/roseanne"
tgt_file = f"nlp_project/workspace/{args.name}.jsonl"

file_paths = [
    os.path.join(base_dir, file) for file in os.listdir(base_dir) if args.pttn in file
]
with open(tgt_file, "w") as f1:
    for file in tqdm(file_paths):
        with open(file, "r") as f:
            content = f.read()
        f1.write(content)
