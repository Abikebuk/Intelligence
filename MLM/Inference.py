import torch
from transformers import RobertaForMaskedLM, AutoTokenizer
from config import default


def inference(model_id, weight_location=default.mlm_model_location, input_text="test input"):
    model = RobertaForMaskedLM.from_pretrained(model_id, load_in_8bit=True)
    print("weight_location")
    print(weight_location)
    weights = torch.load(default.mlm_model_location) \
        if weight_location == "" \
        else torch.load(weight_location)
    print(weights.keys())
    for key in list(weights.keys()):
        weights[key.replace("base_model.model.", "").replace("base_layer.", "")] = weights.pop(key)
    weights = model.load_state_dict(weights, strict=False)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Tokenize the input text
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # Perform inference
    with torch.no_grad():
        outputs = model(input_ids, )
        logits = outputs.logits

        # Generate text
        generated_text = tokenizer.decode(torch.argmax(logits, dim=-1).squeeze())
        print(f"Generated text: {generated_text}")
