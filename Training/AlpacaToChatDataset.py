import json
import random

import torch
import numpy as np
from accelerate.utils import get_data_structure
from torch.utils.data import Dataset
from rich.progress import Progress
from scipy import stats
from tabulate import tabulate

class AlpacaToChatDataset(Dataset):
    """
    Custom dataset for Alpaca to Chat generation converter
    Returns directly tensors for each element of the dataset
    """

    def __init__(self, dataset_path, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        with open(dataset_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def shuffle(self, seed: int=42):
        random.seed(seed)
        random.shuffle(self.data)

    def get_stats(self):
        print("Getting token length stats for current dataset:")
        a = np.array([])
        with Progress() as pbar:
            task = pbar.add_task("[green]Building tokens length stats...", total=len(self.data))
            for e in self.data:
                # Deconstruct the json
                ins = e['instruction']
                inp = e['input']
                out = e['output']
                # Combine data into the expected data structure
                txt = self.get_data_structure(ins, inp, out)
                # Tokenize everything
                tokens = self.tokenizer.encode(txt)
                a = np.append(a, len(tokens))
                pbar.update(task, advance=1)
        a = np.sort(a)
        # Asked claude to give me stats in a table format
        # Calculate statistics
        mean = np.mean(a)
        median = np.median(a)
        std_dev = np.std(a)
        variance = np.var(a)
        min_val = np.min(a)
        max_val = np.max(a)
        range_val = max_val - min_val
        percentiles = np.percentile(a, np.arange(0, 101, 10))
        q1, q3 = np.percentile(a, [25, 75])
        iqr = q3 - q1
        skewness = stats.skew(a)
        kurtosis = stats.kurtosis(a)
        mode = stats.mode(a)
        cv = (std_dev / mean) * 100 if mean != 0 else np.nan

        # Calculate percentiles
        percentiles_range = list(range(0, 91, 10)) + list(range(91, 101))
        percentiles = np.percentile(a, percentiles_range)

        main_stats = [
            ["Mean", mean],
            ["Median", median],
            ["Mode", mode],
            ["Standard Deviation", std_dev],
            ["Variance", variance],
            ["Minimum", min_val],
            ["Maximum", max_val],
            ["Range", range_val],
            ["Interquartile Range (IQR)", iqr],
            ["Skewness", skewness],
            ["Kurtosis", kurtosis],
            ["Coefficient of Variation", f"{cv}%"],
        ]

        # Create and print the main statistics table
        print("Main Statistics:")
        print(tabulate(main_stats, headers=["Statistic", "Value"], tablefmt="pretty"))

        # Prepare data for the percentiles table (vertical layout)
        percentile_stats = [
            [f"{i}th Percentile", p] for i, p in zip(percentiles_range, percentiles)
        ]

        # Create and print the percentiles table
        print("\nPercentiles:")
        print(tabulate(percentile_stats, headers="firstrow", tablefmt="pretty"))

    def get_data_structure(self, ins, inp, out):
        # Convert ins + out into a single string
        txt = ins
        if len(inp) > 0:
            txt += f" with additional input: {inp}."

        # Convert data into expected generation structure
        input_text = f"{{\"role\":\"user\", \"input\":\"{txt}\"}}"
        output_text = f"{{\"output\":\"{out}\"}}"
        return f"{input_text}\n{output_text}\n"

    def __getitem__(self, idx):
        data = self.data[idx]

        # data structure of alpaca dataset is :
        # { "instruction": ins, "input": inp, "output": out }
        ins = data['instruction']
        inp = data['input']
        out = data['output']

        combined_text = self.get_data_structure(ins, inp, out)

        # Convert data into tokens
        input_ids = self.tokenizer.encode(combined_text)

        # Create attention_mask
        attention_mask = [1] * len(input_ids)

        # Generate padding
        if self.max_len > len(input_ids):
            padding_length = self.max_len - len(input_ids)
            input_ids += [self.tokenizer.pad_token_id] * padding_length # padding of pad_token_id
            attention_mask += [0] * padding_length # padding of 0

        # Convert to tensor & trim
        input_ids = torch.tensor(input_ids)[:self.max_len]
        attention_mask = torch.tensor(attention_mask)[:self.max_len]

        # Trim to max_len and return tensors
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids
        }
