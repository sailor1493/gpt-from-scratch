import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score

from .processor import Processor


class NsmcProcessor(Processor):
    num_labels = 2

    def get_tokenized_datasets(self, tokenizer):
        train_data = load_dataset("nsmc", split="train[:90%]")  # 135000
        valid_data = load_dataset("nsmc", split="train[-10%:]") # 15000
        test_data = load_dataset("nsmc", split="test")          # 50000

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

        return train_dataset, valid_dataset, test_dataset

    def compute_metrics(self, p):
        pred, labels = p
        pred = np.argmax(pred, axis=1)
        accuracy = accuracy_score(y_true=labels, y_pred=pred)
        return {"accuracy": accuracy}
