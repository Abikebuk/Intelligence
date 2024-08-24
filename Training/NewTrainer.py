import logging

import deepspeed
import torch
from huggingface_hub import snapshot_download
from peft import get_peft_model
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, SpinnerColumn, \
    MofNCompleteColumn, TimeElapsedColumn
from rich.table import Table
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from .AlpacaToChatDataset import AlpacaToChatDataset
from .Configurator import get_configs


def train(model_path, dataset_path, download_model=False, model_revision=None, batch_size = 1, gradient_accumulation_step = 1, learning_rate = 2e-5, max_length=512, epochs=4, train_ratio=0.8, print_dataset_stats=False, disable_deepspeed_logging=False, limit_dataset_size=None):
    if disable_deepspeed_logging:
        logging.getLogger("DeepSpeed").setLevel(logging.CRITICAL)
    # download model
    model_cache_dir = f"{config.default.hf_models_cache_dir}/{model_path}"
    if download_model:
        snapshot_download(repo_id=model_path, repo_type="model", local_dir=model_cache_dir, revision=model_revision)

    # Configurations
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds_config, lora_config, bnb_config = get_configs(batch_size, gradient_accumulation_step, learning_rate)
    epoch_metrics = []

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_cache_dir, local_files_only=True, quantization_config=bnb_config)
    model = get_peft_model(model, lora_config)
    tokenizer = AutoTokenizer.from_pretrained(model_cache_dir, use_fast=True)

    # Load dataset
    dataset = AlpacaToChatDataset(dataset_path, tokenizer, max_len=max_length, limit_dataset_size=limit_dataset_size)
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
                  TaskProgressColumn(),
                  TextColumn("[bold magenta]{task.speed} steps/s"),
                  TimeElapsedColumn(),
                  TimeRemainingColumn(),
                  transient=False
                  ) as pbar:
        total_task = pbar.add_task("[cyan]Total Training", total=len(dataset)*epochs)
        epoch_task = pbar.add_task("[blue]Epoch", total=len(dataset))
        train_task = pbar.add_task("[purple]Epoch Training", total=len(train_dataloader))
        eval_task = pbar.add_task("[green]Epoch Evaluation...", total=len(eval_dataloader))

        # Train
        for epoch in range(epochs):
            model.train()
            total_train_loss = 0
            total_train_samples = 0
            pbar.reset(epoch_task, description=f"[blue]Epoch {epoch+1}/{epochs}")
            pbar.reset(train_task, visible=True)
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

                # Accumulate loss
                total_train_loss += loss.item() * input_ids.size(0)
                total_train_samples += input_ids.size(0)

                # Progress bar updates
                pbar.update(train_task, advance=1)
                pbar.update(epoch_task, advance=batch_size)
                pbar.update(total_task, advance=batch_size)
            pbar.update(train_task, visible=False)
            # Calculate metrics
            avg_train_loss = total_train_loss / total_train_samples
            train_perplexity = torch.exp(torch.tensor(avg_train_loss))

            # evaluation
            model.eval()
            total_eval_loss = 0
            total_eval_samples = 0
            pbar.reset(eval_task, visible=True)
            for batch in eval_dataloader:
                # Deconstructing batch
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                # Evaluating batch
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                # Accumulate loss
                total_eval_loss += loss.item() * input_ids.size(0)
                total_eval_samples += input_ids.size(0)

                # Progress Bar updates
                pbar.update(eval_task, advance=1)
                pbar.update(epoch_task, advance=batch_size)
                pbar.update(total_task, advance=batch_size)
            pbar.update(eval_task, visible=False)
            # Calculate metrics
            avg_eval_loss = total_eval_loss / total_eval_samples
            eval_perplexity = torch.exp(torch.tensor(avg_eval_loss))

            # Store metrics for this epoch
            epoch_metrics.append({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "train_perplexity": train_perplexity.item(),
                "eval_loss": avg_eval_loss,
                "eval_perplexity": eval_perplexity.item()
            })

    print("Training finished!")
    # Create and display the table
    console = Console()
    table = Table(title="Training and Evaluation Metrics")
    table.add_column("Epoch", justify="right", style="cyan")
    table.add_column("Train Loss", justify="right", style="magenta")
    table.add_column("Train Perplexity", justify="right", style="magenta")
    table.add_column("Eval Loss", justify="right", style="green")
    table.add_column("Eval Perplexity", justify="right", style="green")

    for metrics in epoch_metrics:
        table.add_row(
            str(metrics["epoch"]),
            f"{metrics['train_loss']:.4f}",
            f"{metrics['train_perplexity']:.4f}",
            f"{metrics['eval_loss']:.4f}",
            f"{metrics['eval_perplexity']:.4f}"
        )
    console.print(table)

    print("Saving model...")
    model.save_pretrained("checkpoint")
    print("Saving finished!")