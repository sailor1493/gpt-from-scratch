from abc import ABCMeta, abstractmethod


class Processor(metaclass=ABCMeta):
    @property
    @abstractmethod
    def num_labels(self):
        pass

    @abstractmethod
    def get_tokenized_datasets(self, tokenizer):
        pass

    @abstractmethod
    def compute_metrics(self, p):
        pass
