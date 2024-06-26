import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def train(model_id, dataset_id, eval_ratio: float = 0.2, min_confidence: float = 0.8, batch_size: int = 1,
          learning_rate=5e-5, epochs=2):
    torch.cuda.empty_cache()  # Optionally clear the cache
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    label_filter = [
        "joy",
        "disgust",
        "desire"
    ]
    # LoRA config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    dataset = load_dataset("csv", data_files=dataset_id, cache_dir="cache")['train']
    print(dataset)
    dataset = preprocess_dataset(dataset, label_filter, min_confidence)
    print(dataset.column_names)
    dataset = dataset.rename_column('Predicted Label', 'label')
    columns = dataset.column_names
    for c in columns:
        if c not in label_filter and c != 'Text' and c != 'label':
            dataset = dataset.remove_columns(c)
    print(dataset)
    dataset = tokenize_text(dataset, tokenizer)
    print(dataset)

    # split dataset
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.train_test_split(test_size=eval_ratio)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']
    print(tokenizer.pad_token_id)
    # create dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=collate_fn)

    # configs
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    loss_fn = CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            optimizer.zero_grad()
            input_ids = torch.stack([torch.tensor(item) for item in batch['input_ids']]).to(device)
            attention_mask = torch.stack([torch.tensor(item) for item in batch['attention_mask']]).to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(train_dataloader)

        model.eval()
        eval_loss = 0
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = torch.stack([torch.tensor(item) for item in batch['input_ids']]).to(device)
            attention_mask = torch.stack([torch.tensor(item) for item in batch['attention_mask']]).to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

            eval_loss += loss.item()
        avg_eval_loss = eval_loss / len(eval_dataloader)

    # Save the model
    torch.save(model.state_dict(), "final_model.pth")


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
