import argparse

import numpy as np
from sklearn.metrics import accuracy_score

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from transformers import TrainingArguments, Trainer

def main(args):
    train_data = load_dataset("nsmc", split="train[:90%]")  # 135000
    valid_data = load_dataset("nsmc", split="train[-10%:]") # 15000
    test_data = load_dataset("nsmc", split="test")          # 50000

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # It seems skt/kogpt2 do not has special token in tokenizer
    tokenizer.pad_token = "<pad>"
    tokenizer.unk_token = "<unk>"
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"

    def tokenize_function(example):
        return tokenizer(
            example['document'],
            return_tensors='pt',
            padding=True,
            truncation=True,
            add_special_tokens=True
        )

    train_dataset = train_data.map(tokenize_function, batched=True)
    valid_dataset = valid_data.map(tokenize_function, batched=True)
    test_dataset = test_data.map(tokenize_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir = args.output_dir,
        logging_dir = args.logging_dir,
        num_train_epochs = args.num_train_epochs,
        per_device_train_batch_size = args.train_batch_size,
        per_device_eval_batch_size = args.eval_batch_size,
        logging_steps = args.logging_steps,
        save_steps = args.save_steps,
        save_total_limit = args.save_total_limit
    )

    def compute_metrics(p):
        pred, labels = p
        pred = np.argmax(pred, axis=1)
        accuracy = accuracy_score(y_true=labels, y_pred=pred)
        return {"accuracy": accuracy}

    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    metrics = trainer.evaluate(test_dataset)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()

    cli_parser.add_argument("--model_name_or_path", type=str, required=True)
    cli_parser.add_argument("--num_train_epochs", type=int, required=True)
    cli_parser.add_argument("--output_dir", type=str, default="outputs")
    cli_parser.add_argument("--logging_dir", type=str, default="logs")
    cli_parser.add_argument("--train_batch_size", type=int, default=16)
    cli_parser.add_argument("--eval_batch_size", type=int, default=16)
    cli_parser.add_argument("--logging_steps", type=int, default=1000)
    cli_parser.add_argument("--save_steps", type=int, default=1000)
    cli_parser.add_argument("--save_total_limit", type=int, default=1)
    cli_args = cli_parser.parse_args()

    main(cli_args)
