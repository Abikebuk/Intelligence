from threading import Thread

import deepspeed
import torch
from attr.validators import max_len
from deepspeed.runtime.data_pipeline.data_sampling.indexed_dataset import dtypes
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer, TextIteratorStreamer
from transformers.modeling_utils import unwrap_model


def run_inference(model_id):
    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda:0")
    model = unwrap_model(model)

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model.load_state_dict(torch.load("models/fp32result.bin"), strict=False)
    model.eval()

    print("Chat started. Type your messages and press Enter. Use CTRL+D to exit.")

    conversation_history = []
    while True:
        try:
            user_input = input("> ")

            # Save human input
            conversation_history.append(f"Human: {user_input}")

            full_context = "\n".join(conversation_history) + "\nAI:"

            inputs = tokenizer(full_context, return_tensors="pt").to(device)

            streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)

            # Create a thread to run the generation
            generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=100)
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

            print("AI: ", end="", flush=True)
            generated_text = ""
            for text in streamer:
                print(text, end="", flush=True)
                generated_text += text
            print()  # New line after the complete response

            # Save AI response
            conversation_history.append(f"AI: {generated_text.strip()}")

        except EOFError:  # This is raised when CTRL+D is pressed
            print("\nExiting chat. Goodbye!")
            break
