import torch
from transformers import RobertaForMaskedLM, AutoTokenizer

model = RobertaForMaskedLM.from_pretrained("distilbert/distilroberta-base", load_in_8bit=True)
weights = torch.load("../models/lora_model.pt")
print(weights.keys())
for key in list(weights.keys()):
    weights[key.replace("base_model.model.", "").replace("base_layer.", "")] = weights.pop(key)
weights = model.load_state_dict(weights, strict=False)
model.eval()
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenize the input text
input_text = "The goal of life is <mask>."
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Perform inference
with torch.no_grad():
    outputs = model(input_ids, )
    logits = outputs.logits

    # Generate text
    generated_text = tokenizer.decode(torch.argmax(logits, dim=-1).squeeze(), skip_special_tokens=False)
    print(f"Generated text: {generated_text}")