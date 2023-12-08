import numpy as np
from datasets import load_dataset, load_metric
from sklearn.metrics import accuracy_score
from typing import List, Dict

from .processor import Processor


class KlueNerProcessor(Processor):
    num_labels = 13

    def __init__(self):
        super().__init__()
        self.metric = load_metric("seqeval")
        self.list_of_labels = []

    def get_tokenized_datasets(self, tokenizer):
        train_data = load_dataset("klue", "ner", split="train[:90%]")
        valid_data = load_dataset("klue", "ner", split="train[-10%:]")
        test_data = load_dataset("klue", "ner", split="validation")

        self.list_of_labels = train_data.features["ner_tags"].feature.names

        def relabel_to_token(original_clean_labels, offset_mappings):
            labels_of_tokens = []
            for offset_mapping in offset_mappings:
                cur_start_offset, cur_end_offset = offset_mapping
                if cur_start_offset == cur_end_offset:
                    labels_of_tokens.append(-100)
                    continue
                labels_of_tokens.append(original_clean_labels[cur_start_offset])
            return labels_of_tokens

        def tokenize_function(example):
            original_clean_sentence = "".join(example["tokens"]).replace("\xa0"," ")
            original_clean_labels = example["ner_tags"]

            encoded = tokenizer(original_clean_sentence, return_offsets_mapping=True, return_attention_mask=True, return_token_type_ids=True, add_special_tokens=True, padding=False, truncation=False)
            labels = relabel_to_token(original_clean_labels, encoded["offset_mapping"])
            encoded.update({"labels": labels})
            return encoded

        train_dataset = train_data.map(tokenize_function, remove_columns=["sentence"], batched=False)
        valid_dataset = valid_data.map(tokenize_function, remove_columns=["sentence"], batched=False)
        test_dataset = test_data.map(tokenize_function, remove_columns=["sentence"], batched=False)

        return train_dataset, valid_dataset, test_dataset

    def compute_metrics(self, p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [self.list_of_labels[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.list_of_labels[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = self.metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
