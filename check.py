import os
from tqdm import tqdm
import json


filepath = "/home/n2/chanwoo/nlp-project/gpt-from-scratch/corpus_construction/nlp_project/experiment_corpus/newspaper.json"
tgt = "/home/n2/chanwoo/nlp-project/gpt-from-scratch/corpus_construction/nlp_project/experiment_corpus/newspaper_fixed.json"

key_set = set()
with open(filepath, "r") as f, open(tgt, "w") as tgt_f:
    for line in tqdm(f):
        entry = json.loads(line)
        text = entry["text"]
        uri = entry["uri"]
        doctype = entry["type"]
        payload = {"text": text, "uri": uri, "type": doctype}
        tgt_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
