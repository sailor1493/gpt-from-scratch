from argparse import ArgumentParser
from transformers import AutoTokenizer
from datasets import load_dataset

parser = ArgumentParser()
parser.add_argument("--data-file", type=str, required=True, help="Data file to train")
parser.add_argument(
    "--tokenizer-name", type=str, required=True, help="Tokenizer name to save"
)
args = parser.parse_args()

dataset = load_dataset("json", data_files=args.data_file)

old_tokenizer = AutoTokenizer.from_pretrained("gpt2")

corpus = (x["text"] for x in dataset["train"])

tokenizer = old_tokenizer.train_new_from_iterator(corpus, 52000)
tokenizer.save_pretrained(args.tokenizer_name)
