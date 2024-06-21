import os
import pickle

import numpy as np
import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm


class MLM(Dataset):
    def __init__(self, dataset_id: str = "alpindale/light-novels",
                 model_id: str = "bert-base-uncased",
                 masking_prob: float = 0.15,
                 dataset_range: int = -1,
                 max_length: int = 512):
        self.dataset = load_dataset(dataset_id, split='train').to_pandas()
        if dataset_range != -1:
            self.dataset = self.dataset.head(dataset_range)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, max_length=max_length, truncation=True, padding=True)
        self.masking_prob = masking_prob
        self.max_length = max_length
        self.tokenized_dataset_path = "../tokenized_dataset.pkl.save"
        self.load_or_tokenize_dataset()

    def load_or_tokenize_dataset(self):
        if os.path.exists(self.tokenized_dataset_path):
            print("Loading tokenized dataset from file...")
            with open(self.tokenized_dataset_path, "rb") as f:
                self.dataset["tokens"] = pickle.load(f)
        else:
            print("Tokenizing dataset...")
            self.tokenize_dataset()
            with open(self.tokenized_dataset_path, "wb") as f:
                pickle.dump(self.dataset["tokens"], f)

    def tokenize_dataset(self):
        # self.dataset["text"].apply(self.tokenizer.tokenize)
        self.dataset["tokens"] = [self.tokenizer.tokenize(text[:self.max_length - 1]) for text in tqdm(self.dataset["text"], desc="Tokenizing dataset")]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        tokens = self.dataset["tokens"][idx]
        mask_indices = [i for i, _ in enumerate(tokens) if np.random.rand() < self.masking_prob]
        masked_tokens = [token if i not in mask_indices else "[MASK]" for i, token in enumerate(tokens)]
        return {
            "input_ids": self.tokenizer.convert_tokens_to_ids(masked_tokens),
            "attention_mask": [1] * len(masked_tokens),
            "labels": self.tokenizer.convert_tokens_to_ids(tokens)
        }


def collate_fn(batch):
    # Sort the batch based on the sequence length
    batch = sorted(batch, key=lambda x: len(x["input_ids"]), reverse=True)

    input_ids = [torch.tensor(item["input_ids"]) for item in batch]
    attention_mask = [torch.tensor(item["attention_mask"]) for item in batch]
    labels = [torch.tensor(item["labels"]) for item in batch]

    # Pad sequences to the same length
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
