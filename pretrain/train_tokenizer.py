from argparse import ArgumentParser
from transformers import AutoTokenizer
from datasets import load_dataset
from tokenizers import (
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
)

parser = ArgumentParser()
parser.add_argument("--data-file", type=str, required=True, help="Data file to train")
parser.add_argument(
    "--tokenizer-name", type=str, required=True, help="Tokenizer name to save"
)
args = parser.parse_args()

dataset = load_dataset("json", data_files=args.data_file)

old_tokenizer = AutoTokenizer.from_pretrained("gpt2")

corpus = (x["text"] for x in dataset["train"])


tokenizer = Tokenizer(
    models.BPE(
        bos_token="<|endoftext|>", eos_token="<|endoftext|>", unk_token="<|endoftext|>"
    )
)
normalizer = normalizers.Sequence(
    [
        normalizers.NFKC(),
        normalizers.BertNormalizer(
            clean_text=False,
            handle_chinese_chars=False,
            strip_accents=False,
            lowercase=False,
        ),
    ]
)
tokenizer.normalizer = normalizer

tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
tokenizer.decoder = decoders.Metaspace()
trainer = trainers.BpeTrainer(vocab_size=51200)

tokenizer.train_from_iterator(corpus, trainer=trainer)
tokenizer.save(args.tokenizer_name)