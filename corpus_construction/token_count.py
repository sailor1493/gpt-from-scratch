from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from itertools import chain
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--data-file", type=str, required=True, help="Data file to train")
parser.add_argument(
    "--tokenizer-name", type=str, required=True, help="Tokenizer name to save"
)
parser.add_argument("--log-file", type=str, default="log.txt")
args = parser.parse_args()

data_path = args.data_file
dataset = load_dataset("json", data_files=data_path)["train"]
keys = list(dataset[0].keys())
allowed_keys = ["text"]
for key in keys:
    if key not in allowed_keys:
        dataset = dataset.remove_columns(key)

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)


def tokenize_function(examples):
    output = tokenizer(examples["text"])
    return output


model = AutoModelForCausalLM.from_pretrained("skt/kogpt2-base-v2")
block_size = model.config.n_ctx  # 1024


def group_texts(examples):
    # Concatenate all texts.
    keys = ["input_ids", "attention_mask"]
    concatenated_examples = {k: list(chain(*examples[k])) for k in keys}
    total_length = len(concatenated_examples[keys[0]])
    # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


tokenized_dataset = dataset.map(
    tokenize_function, batched=True, num_proc=80, remove_columns=["text"]
)

lm_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=80)
length = len(lm_dataset["input_ids"])
all_token_count = length * block_size
print(f"Total token count: {all_token_count:,}")

msg = f"Total token count: {all_token_count:,}"
msg += f"\nTotal line count: {length:,}"

with open(args.log_path, "w") as f:
    f.write(msg)
