import random
from tqdm import tqdm
from argparse import ArgumentParser
import os

parser = ArgumentParser()
parser.add_argument("--src", type=str, required=True)
parser.add_argument("--tgt", type=str, required=True)
args = parser.parse_args()

basename = os.path.basename(args.src)

print("=" * 80)
print("Source corpus: {}".format(args.src))
print("Target corpus: {}".format(args.tgt))
print("Basename: {}".format(basename))
print("=" * 80)

print("Shuffling corpus...")
with open(args.src, "r") as f1, open(args.tgt, "w") as f2:
    print("Reading lines...")
    lines = f1.readlines()
    print("Shuffling lines...")
    random.shuffle(lines)
    print("Writing lines...")
    for line in tqdm(lines):
        f2.write(line)
