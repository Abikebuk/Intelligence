import deepspeed
import torch
from attr.validators import max_len
from deepspeed.runtime.data_pipeline.data_sampling.indexed_dataset import dtypes
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer
from transformers.modeling_utils import unwrap_model


def run_inference(model_id):
    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda:0")
    model = unwrap_model(model)

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model.load_state_dict(torch.load("models/fp32result.bin"), strict=False)
    model.eval()


    with torch.inference_mode():
        tensor = tokenizer("I am", return_tensors="pt").input_ids.to(device)
        output = model.generate(tensor, max_length=512)
        decoded_output = tokenizer.decode(output[0])
        print(decoded_output)

