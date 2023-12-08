import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from typing import List, Dict

from .processor import Processor


class KlueNliProcessor(Processor):
    num_labels = 3

    def get_tokenized_datasets(self, tokenizer):
        train_data = load_dataset("klue", "nli", split="train[:90%]")
        valid_data = load_dataset("klue", "nli", split="train[-10%:]")
        test_data = load_dataset("klue", "nli", split="validation")

        def tokenize_function(example):
            hypotheses = example['hypothesis']
            premises = example['premise']

            dicf_of_training_examples: Dict[str, List[List[int]]] = {}

            for hypothesis, premise in zip(hypotheses, premises):
                list_of_tokens_from_hypothesis = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(hypothesis))
                list_of_tokens_from_premise = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(premise))
                training_example = tokenizer.prepare_for_model(
                    list_of_tokens_from_premise,
                    list_of_tokens_from_hypothesis,
                    add_special_tokens=True,
                    padding=False,
                    truncation=False
                )

                for key in training_example.keys():
                    if key not in dicf_of_training_examples:
                        dicf_of_training_examples.setdefault(key, [])
                    dicf_of_training_examples[key].append(training_example[key])

            return dicf_of_training_examples

        train_dataset = train_data.map(tokenize_function, batched=True).rename_column("label", "labels")
        valid_dataset = valid_data.map(tokenize_function, batched=True).rename_column("label", "labels")
        test_dataset = test_data.map(tokenize_function, batched=True).rename_column("label", "labels")

        return train_dataset, valid_dataset, test_dataset

    def compute_metrics(self, p):
        pred, labels = p
        pred = np.argmax(pred, axis=1)
        accuracy = accuracy_score(y_true=labels, y_pred=pred)
        return {"accuracy": accuracy}
