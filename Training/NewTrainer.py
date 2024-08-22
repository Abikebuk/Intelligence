import logging

import deepspeed
import torch
from exllamav2 import ExLlamaV2Config, ExLlamaV2, ExLlamaV2Tokenizer
from huggingface_hub import hf_hub_download, snapshot_download
from peft import get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, SpinnerColumn, \
    MofNCompleteColumn, TimeElapsedColumn, RenderableColumn, TransferSpeedColumn

import config
import utils
from .AlpacaToChatDataset import AlpacaToChatDataset
from .Configurator import get_configs


def train(model_path, dataset_path, download_model=False, model_revision=None, batch_size = 1, gradient_accumulation_step = 1, learning_rate = 2e-5, max_length=512, epochs=4, train_ratio=0.8, print_dataset_stats=False, disable_deepspeed_logging=False):
    if disable_deepspeed_logging:
        logging.getLogger("DeepSpeed").setLevel(logging.CRITICAL)
    # download model
    model_cache_dir = f"{config.default.hf_models_cache_dir}/{model_path}"
    if download_model:
        snapshot_download(repo_id=model_path, repo_type="model", local_dir=model_cache_dir, revision=model_revision)

    # Configurations
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds_config, lora_config, bnb_config = get_configs(batch_size, gradient_accumulation_step, learning_rate)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_cache_dir, local_files_only=True, quantization_config=bnb_config)
    model = get_peft_model(model, lora_config)
    tokenizer = AutoTokenizer.from_pretrained(model_cache_dir, use_fast=True)

    # Load dataset
    dataset = AlpacaToChatDataset(dataset_path, tokenizer, max_len=max_length)
    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)

    dataset.get_stats()

    # deepspeed config
    model, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config
    )

    #
    torch.cuda.empty_cache()

    # Progress init
    with Progress(TextColumn("[progress.description]{task.description}"),
                  SpinnerColumn(),
                  BarColumn(),
                  MofNCompleteColumn(),
                  TransferSpeedColumn(),
                  TaskProgressColumn(),
                  TextColumn("[bold magenta]{task.speed} steps/s"),
                  TimeElapsedColumn(),
                  TimeRemainingColumn(),
                  ) as pbar:
        training_task = pbar.add_task("[cyan]Total Training", total=len(dataset)*epochs)
        epoch_task = pbar.add_task("[blue]Total Epoch", total=len(dataset))
        # Train
        for epoch in range(epochs):
            batch_task = pbar.add_task("[purple]Epoch Training", total=len(train_dataloader))
            model.train()
            for batch in train_dataloader:
                # Deconstructing batch
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                # Training batch
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                model.backward(loss)
                model.step()
                # Progress bar updates
                pbar.update(batch_task, advance=1)
                pbar.update(epoch_task, advance=batch_size)
                pbar.update(training_task, advance=batch_size)
            pbar.stop_task(batch_task)

            # evaluation
            model.eval()
            eval_task = pbar.add_task("[green]Epoch Evaluation...", total=len(eval_dataloader))
            for batch in eval_dataloader:
                print(batch)
                pbar.update(eval_task, advanvce=1)
                pbar.update(epoch_task, advance=batch_size)
                pbar.update(training_task, advance=batch_size)

            pbar.stop_task(eval_task)

