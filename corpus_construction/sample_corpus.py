from tqdm import tqdm
import os
import random
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--src", type=str, required=True)
parser.add_argument("--tgt", type=str, required=True)
parser.add_argument("--ratio", type=float, default=0.1)

args = parser.parse_args()

basename = os.path.basename(args.src)

print("=" * 80)
print("Source corpus: {}".format(args.src))
print("Target corpus: {}".format(args.tgt))
print("Basename: {}".format(basename))
print("=" * 80)

print("Sampling corpus...")
with open(args.src, "r") as f1, open(args.tgt, "w") as f2:
    for line in tqdm(f1):
        if random.random() < args.ratio:
            f2.write(line)
