import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset


# load model
device="cuda"
"""
baseline: [
            "skt/kogpt2-base-v2", 
            "/data/s1/chanwoo/nlp_project/logs/bulk_books", 
            "/data/s1/chanwoo/nlp_project/logs/specialized", 
            "/data/s1/chanwoo/nlp_project/logs/newspaper", 
            "/data/s1/chanwoo/nlp_project/logs/mixup"
          ]`
"""
model_path = "skt/kogpt2-base-v2" 
model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path,
  bos_token='<s>', eos_token='</s>', unk_token='<unk>',
  pad_token='<pad>', mask_token='<mask>')

# load dataset
data="NIKL_WRITTEN_v1.2"
print(">>> Loading Dataset...")
dataset = load_dataset("json", data_files="/data/s1/parsed_corpus/NIKL/" + data + ".jsonl", split='train')
print(">>> Dataset Length:", len(dataset['text']))

# tokenize
print(">>> Tokenizing...")
encodings = tokenizer("\n\n".join(dataset['text'][:1]), return_tensors="pt")

# evaluate perplexity
print(">>> Evaluating Perplexity...")
max_length = model.config.n_positions
stride = 512
seq_len = encodings.input_ids.size(1)

nlls = []
prev_end_loc = 0
for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)

        # loss is calculated using CrossEntropyLoss which averages over valid labels
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
        neg_log_likelihood = outputs.loss

    nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).mean())
print(">>> Perplexity:", ppl.item())