import argparse

from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer
from transformers import DataCollatorWithPadding, DataCollatorForTokenClassification
from transformers import TrainingArguments, Trainer

from processor import NsmcProcessor, KlueNliProcessor, KlueNerProcessor


def main(args):
    processor = None
    if args.task == 'nsmc':
        processor = NsmcProcessor()
    elif args.task == 'klue_nli':
        processor = KlueNliProcessor()
    elif args.task == 'klue_ner':
        processor = KlueNerProcessor()

    model = None
    if args.task == 'nsmc' or args.task == 'klue_nli':
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=processor.num_labels)
    elif args.task == 'klue_ner':
        model = AutoModelForTokenClassification.from_pretrained(args.model_name_or_path, num_labels=processor.num_labels)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # skt/kogpt2 do not has special token in tokenizer
    tokenizer.pad_token = "<pad>"
    tokenizer.unk_token = "<unk>"
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"

    train_dataset, valid_dataset, test_dataset = processor.get_tokenized_datasets(tokenizer=tokenizer)

    data_collator = None
    if args.task == 'nsmc' or args.task == 'klue_nli':
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    elif args.task == 'klue_ner':
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)

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

    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=processor.compute_metrics
    )

    trainer.train()

    metrics = trainer.evaluate(test_dataset)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()

    cli_parser.add_argument("--task", type=str, required=True)
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
