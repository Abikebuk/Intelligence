import torch
from peft import LoraConfig
from transformers import BitsAndBytesConfig


def get_configs(batch_size = 12, gradient_accumulation_step = 4, learning_rate = 2e-5):
    # DeepSpeed configuration
    ds_config = {
        "verbose": False,
        "train_batch_size": batch_size * gradient_accumulation_step,
        "bf16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": "cpu",
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

    # LoRA config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        load_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    return ds_config, lora_config, bnb_config
