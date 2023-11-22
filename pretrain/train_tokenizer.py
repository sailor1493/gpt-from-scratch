from argparse import ArgumentParser
from datasets import load_dataset
from tokenizers import (
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
    AddedToken,
)
from transformers import AutoTokenizer

parser = ArgumentParser()
parser.add_argument("--data-file", type=str, required=True, help="Data file to train")
parser.add_argument(
    "--tokenizer-name", type=str, required=True, help="Tokenizer name to save"
)
args = parser.parse_args()

dataset = load_dataset("json", data_files=args.data_file)
corpus = (x["text"] for x in dataset["train"])

to_add = AddedToken(
    "<|endoftext|>", single_word=False, lstrip=False, rstrip=False, normalized=True
)
kogpt_tokenizer = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2")
new_tokenizer = kogpt_tokenizer.train_new_from_iterator(
    corpus, kogpt_tokenizer.vocab_size
)
new_tokenizer.save_pretrained(f"../tokenizers/{args.tokenizer_name}")
