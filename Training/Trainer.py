import logging
import statistics
from pathlib import Path

import deepspeed
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from progress_table import ProgressTable
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import config
from .CheckpointSampler import dataloader_to_step


def train(model_id, dataset_id, eval_ratio: float = 0.2, min_confidence: float = 0.8, batch_size: int = 16,
          learning_rate=2e-5, epochs=4, label_filter=None, model_output=config.default.trainer_output_location,
          gradient_accumulation_step=1, clear_cache_every_x_batches: int = 100, checkpoint_save_interval: int = None,
          checkpoint_dir="checkpoint", padding_max_size=100, limit_dataset_size=None):
    # TODO: Checkpoints don't work well
    # Initialization
    torch.cuda.empty_cache()
    logging.getLogger("deepspeed").setLevel(logging.WARNING)
    # Progress table initialization
    table = ProgressTable(
        pbar_embedded=False,
        pbar_style="angled alt red blue",
        pbar_show_progress=True,
        pbar_show_throughput=True,
        pbar_show_eta=True,
        default_header_color="bold",
    )
    table.add_column("Epoch", width=6)
    table.add_column("Step", width=10)
    table.add_column("Batch", width=20)
    table.add_column("Loss", width=20)
    table.add_column("Average Loss", width=20)

    # Create configs required for the training
    print("Creating configs...")
    ds_config, bnb_config, lora_config = create_config(batch_size, gradient_accumulation_step, learning_rate)

    print("Loading Model...")
    model = AutoModelForCausalLM.from_pretrained(model_id, use_cache=False,
                                                 quantization_config=bnb_config)  #.to(device)
    model = get_peft_model(model, lora_config)  ##.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    print("Loading dataset...")
    dataset = load_dataset("csv", data_files=dataset_id, cache_dir="cache")['train']

    # Filter the dataset
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

    # Print the tokens stats from the dataset
    # get_token_length_stats(dataset)

    # Split dataset
    print("Splitting dataset...")
    dataset = dataset.shuffle(seed=42)
    if limit_dataset_size is not None and limit_dataset_size < len(dataset):
        dataset = dataset.select(range(limit_dataset_size))
        print("Limiting dataset size to {}".format(limit_dataset_size))
    dataset = dataset.train_test_split(test_size=eval_ratio)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']

    # Create dataloader
    print("Creating Dataloaders...")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  collate_fn=lambda a: collate_fn(a, tokenizer, padding_max_size))
    total_train_len = len(train_dataloader)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size,
                                 collate_fn=lambda a: collate_fn(a, tokenizer, padding_max_size))

    # Optimization steps
    torch.compile(model)

    # Initialize DeepSpeed engine
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )
    deepspeed.init_distributed()
    _, training_state = model_engine.load_checkpoint(checkpoint_dir)
    training_state = {'step': 0, 'epoch': 0, 'total_loss': 0} if training_state is None else training_state
    step = training_state['step']
    begin_from_checkpoint = False
    looped_once = False
    if step != 0:
        begin_from_checkpoint = True
    train_dataloader = dataloader_to_step(train_dataloader, step + 1)

    # Training
    print("Training model...")
    for epoch in range(epochs):
        table.update("Step", "Training")
        model_engine.train()
        total_loss = 0
        if begin_from_checkpoint:
            if looped_once:
                train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                              collate_fn=lambda a: collate_fn(a, tokenizer))
                begin_from_checkpoint = False
            else:
                epoch = training_state['epoch']
                total_loss = training_state['total_loss']
        table.update("Epoch", f"{epoch + 1}/{epochs}")
        for i, batch in enumerate(table(train_dataloader, total=len(train_dataloader))):
            table.update("Batch", f"{i + 1}/{len(train_dataloader)}")

            input_ids = torch.stack([item.clone().detach() for item in batch['input_ids']])
            attention_mask = torch.stack([item.clone().detach() for item in batch['attention_mask']])

            outputs = model_engine(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

            model_engine.backward(loss)
            model_engine.step()

            loss = loss.item()
            total_loss += loss

            table.update("Loss", loss)
            table.update("Average Loss", total_loss / (i + 1))
            # Checkpoint saving
            if checkpoint_save_interval is not None and i != 0 and i % checkpoint_save_interval == 0:
                training_state['step'] = i
                training_state['total_loss'] = total_loss
                model_engine.save_checkpoint(checkpoint_dir, client_state=training_state)
                table.next_row()
            del input_ids, attention_mask
            if i % clear_cache_every_x_batches == 0:
                torch.cuda.empty_cache()

        avg_loss = total_loss / total_train_len
        print(f"Average loss: {avg_loss:.4f}")

        # Evaluation
        table.update("Step", "Evaluation")
        table.next_row()
        model_engine.eval()
        eval_loss = 0
        for i, batch in enumerate(table(eval_dataloader, total=len(eval_dataloader))):
            table.update("Batch", f"Batch {i + 1}/{len(eval_dataloader)}")
            input_ids = torch.stack([item.clone().detach() for item in batch['input_ids']])
            attention_mask = torch.stack([item.clone().detach() for item in batch['attention_mask']])

            with torch.no_grad():
                outputs = model_engine(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss

            model_engine.step()
            loss = loss.item()
            eval_loss += loss

            table.update("Loss", loss)
            table.update("Average Loss", eval_loss / (i + 1))

        avg_eval_loss = eval_loss / len(eval_dataloader)
        print(f"Average evaluation loss: {avg_eval_loss:.4f}")

        # Checkpoint after evaluation
        model_engine.save_checkpoint(checkpoint_dir)
        looped_once = True

    # Saving model
    output = Path(model_output)
    model_engine.save_16bit_model(output.parent, output.name)


def preprocess_dataset(dataset, label_filter, min_confidence):
    def gte_confidence(row):
        predicted_label = row["Predicted Label"]
        confidence_value = row[predicted_label]
        return isinstance(row['Text'], str) and predicted_label in label_filter and confidence_value >= min_confidence

    return dataset.filter(gte_confidence)


def shuffle_dataset(dataset):
    return dataset.shuffle(seed=42)


def tokenize_text(dataset, tokenizer):
    def p(row, idx):
        return tokenizer(row['Text'], truncation=True)

    return dataset.map(p, batched=True, with_indices=True)


def collate_fn(batch, tokenizer, padding_max_size=100):
    input_ids = [torch.tensor(item['input_ids'], dtype=torch.long) for item in batch]
    attention_mask = [torch.tensor(item['attention_mask'], dtype=torch.long) for item in batch]
    max_len = max(seq.size(0) for seq in input_ids)

    if max_len > padding_max_size:
        # If max_size is provided, use it as the padding size
        max_len = padding_max_size

    # Pad sequences to max_len
    input_ids_padded = torch.stack([
        torch.nn.functional.pad(seq, (0, max_len - seq.size(0)), value=tokenizer.pad_token_id)
        for seq in input_ids
    ])
    attention_mask_padded = torch.stack([
        torch.nn.functional.pad(seq, (0, max_len - seq.size(0)), value=0)
        for seq in attention_mask
    ])

    return {
        'input_ids': input_ids_padded.pin_memory(),
        'attention_mask': attention_mask_padded.pin_memory()
    }


def create_config(batch_size, gradient_accumulation_step, learning_rate):
    # DeepSpeed configuration
    ds_config = {
        "train_batch_size": batch_size * gradient_accumulation_step,
        "bf16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": "nvme",
                "pin_memory": True
            },
            "offload_param": {
                "device": "nvme",
                "pin_memory": True
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True
        },
        "gradient_accumulation_steps": gradient_accumulation_step,
        "gradient_clipping": 1.0,
        "steps_per_print": 2000,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": learning_rate,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01,
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": learning_rate,
                "warmup_num_steps": 1000
            }
        },
        "comms_logger": {
            "enabled": False,
            "verbose": False,
            "prof_all": False
        },
        "quantize_training": {
            "enabled": True,
            "quantize_verbose": True,
            "quantizer_kernel": True,
            "quantize_type": "symmetric",
            "quantize_bits": {
                "start_bits": 16,
                "target_bits": 4
            },
            "quantize_schedule": {
                "quantize_period": 10,
                "schedule_offset": 0
            },
            "quantize_groups": 8,
            "fp16_mixed_quantize": {
                "enabled": True,
                "quantize_change_ratio": 0.001
            },
            "eigenvalue": {
                "enabled": True,
                "verbose": True,
                "max_iter": 50,
                "tol": 1e-2,
                "stability": 0,
                "gas_boundary_resolution": 1,
                "layer_name": "bert.encoder.layer",
                "layer_num": 12
            }
        }
    }

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        load_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # LoRA config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    return ds_config, bnb_config, lora_config
