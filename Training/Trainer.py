import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.cuda.amp import autocast, GradScaler

import config
import utils


def train(model_id, dataset_id, eval_ratio: float = 0.2, min_confidence: float = 0.8, batch_size: int = 16,
          learning_rate=1e-5, epochs=4, label_filter=None, model_output=config.default.trainer_output_location,
          gradient_accumulation_step = 2):
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(model_id, use_cache=False)#.to(device)
    # LoRA config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)##.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    dataset = load_dataset("csv", data_files=dataset_id, cache_dir="cache")['train']

    # filter
    if label_filter is None or len(label_filter) == 0:
        label_filter = dataset.features
    print("Dataset infos:")
    print(dataset)
    print()
    print("Preprocessing the dataset. Applying filter and minimum confidence...")
    dataset = preprocess_dataset(dataset, label_filter, min_confidence)
    dataset = dataset.rename_column('Predicted Label', 'label')
    columns = dataset.column_names

    # Remove columns not in label_filter
    for c in columns:
        if c not in label_filter and c != 'Text' and c != 'label':
            dataset = dataset.remove_columns(c)
    print("Dataset infos after preprocessing:")
    print(dataset)
    print()

    print("Tokenizing dataset...")
    dataset = tokenize_text(dataset, tokenizer)

    # split dataset
    print("Splitting dataset...")
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.train_test_split(test_size=eval_ratio)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']


    # create dataloader
    print("Creating Dataloaders...")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=collate_fn)


    # configs
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scaler = GradScaler()

    # training
    print("Training model...")
    for epoch in range(epochs):
        # Training dataset
        model.train()
        total_loss = 0
        for i, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}")):
            input_ids = torch.stack([item.clone().detach() for item in batch['input_ids']])#.to(device)
            attention_mask = torch.stack([item.clone().detach()  for item in batch['attention_mask']])#.to(device)

            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss

            scaler.scale(loss).backward()

            # gradient accumulation
            if (i + 1) % gradient_accumulation_step == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * gradient_accumulation_step
        avg_loss = total_loss / len(train_dataloader)
        print(f"Average loss: {avg_loss:.4f}'")

        # Evaluating dataset
        model.eval()
        eval_loss = 0
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = torch.stack([item.clone().detach() for item in batch['input_ids']])#.to(device)
            attention_mask = torch.stack([item.clone().detach() for item in batch['attention_mask']])#.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

            eval_loss += loss.item()
        avg_eval_loss = eval_loss / len(eval_dataloader)
        print(f"Average evaluation loss: {avg_eval_loss:.4f}'")

        # Save the model
        utils.create_dirs(model_output)
        torch.save(model.state_dict(), model_output)

def preprocess_dataset(dataset, label_filter, min_confidence):
    def gte_confidence(row):
        predicted_label = row["Predicted Label"]
        confidence_value = row[predicted_label]
        return isinstance(row['Text'], str) and predicted_label in label_filter and confidence_value >= min_confidence

    return dataset.filter(gte_confidence)

def shuffle_dataset(dataset, eval_ratio):
    return dataset.shuffle(seed=42)

def tokenize_text(dataset, tokenizer):
    def p(row, idx):
        return tokenizer(row['Text'], truncation=True, max_length=512)
    return dataset.map(p, batched=True, with_indices=True)

def collate_fn(batch):
    input_ids = [torch.tensor(item['input_ids']) for item in batch]
    attention_mask = [torch.tensor(item['attention_mask']) for item in batch]
    input_ids_padded = pad_sequence(input_ids, batch_first=True)
    attention_mask_padded = pad_sequence(attention_mask, batch_first=True)
    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_mask_padded
    }
