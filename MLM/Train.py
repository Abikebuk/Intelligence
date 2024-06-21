from datetime import datetime, timedelta
import torch
from peft import get_peft_model
from peft.tuners import lora
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, RobertaForMaskedLM
from tqdm import tqdm
from .Dataset import MLMDateset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model_id,
          dataset_id,
          num_epochs=3,
          batch_size=16,
          max_dataset_size=-1,
          tokens_path="",
          model_path="models/mlm_model.pt"):
    # Actual mlm
    print("Preparing dataset with Masked Language Modeling...")
    mlm = MLMDateset(
        dataset_id=dataset_id,
        model_id=model_id,
        masking_prob=0.15,
        dataset_range=max_dataset_size,
        tokens_path=tokens_path)
    print("Loading dataloader...")
    dataloader = DataLoader(mlm, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Optimizer and scheduler
    compute_dtype = getattr(torch, "float16")

    print("Loading model ...")
    model = RobertaForMaskedLM.from_pretrained(model_id, load_in_8bit=True)
    print(model)
    # Initialize LoRA
    print("Initializing LoRA...")
    lora_config = lora.LoraConfig(
        r=8,  # Rank of the low-rank decomposition
        lora_alpha=16,  # Scale for the low-rank decomposition
        target_modules=["query", "value"],  # Modules to apply LoRA to,
    )
    lora_model = get_peft_model(model, lora_config, "default")

    # optimizer = AdamW(model.parameters(), lr=2e-5)
    print("Initializing training parameters...")
    optimizer = AdamW(lora_model.parameters(), lr=2e-5)
    num_training_steps = len(dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # Training loop
    print("Starting training...")
    start_time = datetime.now()
    for epoch in range(num_epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch", dynamic_ncols=True)
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            # output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            output = lora_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = output.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Time estimation + progress bar update
            current_time = datetime.now()
            time_elapsed = current_time - start_time
            time_per_epoch = time_elapsed / (epoch + 1)
            estimated_finish_time = start_time + timedelta(
                seconds=time_per_epoch.total_seconds() * (num_epochs - epoch - 1))
            progress_bar.set_postfix(estimated_finish_time=estimated_finish_time.strftime("%Y-%m-%d %H:%M:%S"))

    torch.save(lora_model.state_dict(), "models/lora_model.pt")


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
