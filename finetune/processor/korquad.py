from datasets import load_dataset
from .processor import Processor
from tqdm.auto import tqdm
import collections
import numpy as np
import evaluate

class KorquadProcessor(Processor):
    num_labels = None
    
    def __init__(self):
        super().__init__()
        self.metric = evaluate.load("squad")
        self.max_length = 384
        self.stride = 128
        self.n_best = 15
        self.max_answer_length = 30
        

    def get_tokenized_datasets(self, tokenizer):
        raw_datasets = load_dataset("KETI-AIR/korquad", 'v1.0')
        raw_datasets['validation'] = raw_datasets.pop('dev')

        def preprocess_training_examples(examples):
            questions = [q.strip() for q in examples["question"]]
            inputs = tokenizer(
                questions,
                examples["context"],
                max_length=self.max_length,
                truncation="only_second",
                stride=self.stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
            )

            offset_mapping = inputs.pop("offset_mapping")
            sample_map = inputs.pop("overflow_to_sample_mapping")
            answers = examples["answers"]
            start_positions = []
            end_positions = []

            for i, offset in enumerate(offset_mapping):
                sample_idx = sample_map[i]
                answer = answers[sample_idx]
                start_char = answer["answer_start"][0]
                end_char = answer["answer_start"][0] + len(answer["text"][0])
                sequence_ids = inputs.sequence_ids(i)

                # Find the start and end of the context
                idx = 0
                while sequence_ids[idx] == 0:
                    idx += 1
                context_start = idx
                #while sequence_ids[idx] == 1:
                while idx != len(sequence_ids):
                    idx += 1
                context_end = idx - 1

                # If the answer is not fully inside the context, label is (0, 0)
                if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                    start_positions.append(0)
                    end_positions.append(0)
                else:
                    # Otherwise it's the start and end token positions
                    idx = context_start
                    while idx <= context_end and offset[idx][0] <= start_char:
                        idx += 1
                    start_positions.append(idx - 1)

                    idx = context_end
                    while idx >= context_start and offset[idx][1] >= end_char:
                        idx -= 1
                    end_positions.append(idx + 1)

            inputs["start_positions"] = start_positions
            inputs["end_positions"] = end_positions
            return inputs
        

        def preprocess_validation_examples(examples):
            questions = [q.strip() for q in examples["question"]]
            inputs = tokenizer(
                questions,
                examples["context"],
                max_length=self.max_length,
                truncation="only_second",
                stride=self.stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
        )

            sample_map = inputs.pop("overflow_to_sample_mapping")
            example_ids = []

            for i in range(len(inputs["input_ids"])):
                sample_idx = sample_map[i]
                example_ids.append(examples["id"][sample_idx])

                sequence_ids = inputs.sequence_ids(i)
                offset = inputs["offset_mapping"][i]
                inputs["offset_mapping"][i] = [
                    o if sequence_ids[k] == 1 or None else None for k, o in enumerate(offset)
                ]

            inputs["example_id"] = example_ids
            return inputs
        
        train_dataset = raw_datasets["train"].map(
            preprocess_training_examples,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
        )   

        valid_dataset = raw_datasets["validation"].map(
           preprocess_validation_examples,
            batched=True,
            remove_columns=raw_datasets["validation"].column_names,
        )

        return raw_datasets, train_dataset, valid_dataset
    

    def compute_metrics(self, start_logits, end_logits, features, examples):
        example_to_features = collections.defaultdict(list)
        for idx, feature in enumerate(features):
            example_to_features[feature["example_id"]].append(idx)

        predicted_answers = []
        for example in tqdm(examples):
            example_id = example["id"]
            context = example["context"]
            answers = []

            # Loop through all features associated with that example
            for feature_index in example_to_features[example_id]:
                start_logit = start_logits[feature_index]
                end_logit = end_logits[feature_index]
                offsets = features[feature_index]["offset_mapping"]

                start_indexes = np.argsort(start_logit)[-1 : -self.n_best - 1 : -1].tolist()
                end_indexes = np.argsort(end_logit)[-1 : -self.n_best - 1 : -1].tolist()
                for start_index in start_indexes:
                    if start_index >= len(offsets):
                            continue
                    for end_index in end_indexes:
                        if end_index >= len(offsets):
                            continue
                        # Skip answers that are not fully in the context
                        if offsets[start_index] is None or offsets[end_index] is None:
                            continue
                        # Skip answers with a length that is either < 0 or > max_answer_length
                        if (
                            end_index < start_index
                            or end_index - start_index + 1 > self.max_answer_length
                        ):
                            continue

                        answer = {
                            "text": context[offsets[start_index][0] : offsets[end_index][1]],
                            "logit_score": start_logit[start_index] + end_logit[end_index],
                        }
                        answers.append(answer)

            # Select the answer with the best score
            if len(answers) > 0:
                best_answer = max(answers, key=lambda x: x["logit_score"])
                predicted_answers.append(
                    {"id": example_id, "prediction_text": best_answer["text"]}
                )
            else:
                predicted_answers.append({"id": example_id, "prediction_text": ""})

        theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
        return self.metric.compute(predictions=predicted_answers, references=theoretical_answers)