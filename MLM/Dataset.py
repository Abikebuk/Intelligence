import os
import pickle
from pathlib import Path
import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

import utils
from config import default


class MLMDateset(Dataset):
    def __init__(self, dataset_id: str,
                 model_id: str,
                 masking_prob: float = 0.15,
                 dataset_range: int = -1,
                 max_length: int = 512,
                 tokens_path: str = "tokens/mlm_dataset.pkl"):
        self.dataset = load_dataset(dataset_id, split='train').to_pandas()
        if dataset_range != -1:
            self.dataset = self.dataset.head(dataset_range)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, max_length=max_length, truncation=True, padding=True)
        self.masking_prob = masking_prob
        self.max_length = max_length
        if tokens_path == "":
            self.tokenized_dataset_path = default.mlm_tokens_location
        else:
            self.tokenized_dataset_path = tokens_path
        self.load_or_tokenize_dataset()

    def load_or_tokenize_dataset(self):
        if os.path.exists(self.tokenized_dataset_path):
            print("Loading tokenized dataset from file...")
            with open(self.tokenized_dataset_path, "rb") as f:
                self.dataset["tokens"] = pickle.load(f)
        else:
            print("Tokenizing dataset...")
            self.tokenize_dataset()
            file = Path(self.tokenized_dataset_path)
            utils.create_dirs(self.tokenized_dataset_path)
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


