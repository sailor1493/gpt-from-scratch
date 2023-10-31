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
    AddedToken,
)
from transformers import PreTrainedTokenizerFast

parser = ArgumentParser()
parser.add_argument("--data-file", type=str, required=True, help="Data file to train")
parser.add_argument(
    "--tokenizer-name", type=str, required=True, help="Tokenizer name to save"
)
args = parser.parse_args()

dataset = load_dataset("json", data_files=args.data_file)

old_tokenizer = AutoTokenizer.from_pretrained("gpt2")

corpus = (x["text"] for x in dataset["train"])

to_add = AddedToken(
    "<|endoftext|>", single_word=False, lstrip=False, rstrip=False, normalized=True
)
tokenizer = Tokenizer(
    models.BPE(),
)
tokenizer.add_special_tokens([to_add])
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
trainer = trainers.BpeTrainer(vocab_size=50257, special_tokens=[to_add])

tokenizer.train_from_iterator(corpus, trainer=trainer)
# tokenizer.save(args.tokenizer_name)


tok_to_push = PreTrainedTokenizerFast(
    model_max_length=1024,
    bos_token=to_add,
    eos_token=to_add,
    unk_token=to_add,
    clean_up_tokenization_spaces=True,
    add_prefix_space=False,
    tokenizer_object=tokenizer,
)
tok_to_push.save_pretrained(args.tokenizer_name)
