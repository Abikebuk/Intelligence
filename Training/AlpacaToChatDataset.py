import json

import torch
from torch.utils.data import Dataset


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

    def __getitem__(self, idx):
        data = self.data[idx]

        # data structure of alpaca dataset is :
        # { "instruction": ins, "input": inp, "output": out }
        ins = data['instruction']
        inp = data['input']
        out = data['output']

        # Convert ins + out into a single string
        txt = ins
        if len(inp) > 0:
            txt += f" with additional input: {inp}."

        # Convert data into expected generation structure
        input_text = f"{{\"role\":\"user\", \"input\":\"{txt}\"}}"
        output_text = f"{{\"output\":\"{out}\"}}"
        combined_text = f"{input_text}\n{output_text}\n"

        # Convert data into tokens
        input_ids = self.tokenizer.encode(combined_text)[0]

        # Get attention_mask
        attention_mask = [1] * len(input_ids)

        # Generate padding
        padding_length = self.max_len - len(input_ids)
        padding =  torch.tensor([self.tokenizer.pad_token_id] * padding_length)
        input_ids = torch.cat((input_ids, padding))
        attention_mask += [0] * padding_length

        # Return
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(input_ids)
        }
