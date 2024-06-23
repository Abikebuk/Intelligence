from asyncio import sleep

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
import utils


def classify(model_id, dataset_id, label_list, dataset_range: int = None, save_location: str = None):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=len(label_list))

    # Sample text data
    text = load_dataset(dataset_id, split="train")
    if dataset_range is not None:
        text = text.select(range(dataset_range))
    text = text["text"]

    # Preprocess text
    # encoded_data = utils.load_pickle(save_location)
    encoded_data = None
    if encoded_data is None:
        print("Tokenizing dataset...")
        encoded_data = tokenizer(
            [text for text in tqdm(text, desc="Tokenizing dataset")],
            padding=True, truncation=True, return_tensors="pt")
        if save_location is not None:
            utils.save_pickle(save_location, encoded_data)

    # Convert to tensors
    input_ids = encoded_data['input_ids'].clone().detach()
    attention_mask = encoded_data['attention_mask'].clone().detach()
    print(input_ids)
    # Feed data to model and get logits
    # outputs = model(input_ids, attention_mask=attention_mask)
    # logits = outputs.logits

    # Get probabilities for each class
    # prediction = torch.argmax(logits, dim=-1)
    # probabilities = torch.softmax(logits, dim=-1)
    # Print predictions
    print(enumerate(text))
    with tqdm(total=len(text), desc="Classifying dataset", leave=True) as pbar:

        i = 0
        for text_data in text:
            outputs = model(input_ids[i], attention_mask=attention_mask[i])
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=-1)
            probabilities = torch.softmax(logits, dim=-1)
            print(f"Text: {text_data}")
            # Sort probabilities in descending order
            # Print the predicted label
            predicted_label = label_list[prediction[i].item()]
            # print(f"  Predicted Label: {predicted_label}")
            # Print the top 5 probabilities for the predicted label in descending order
            sorted_probs, sorted_indices = torch.sort(probabilities[i], descending=True)
            # Print only the top 5 probabilities in one line
            # top_5_probs = [f"{label_list[sorted_indices[j]]}: {sorted_probs[j].item():.4f}" for j in range(5)]
            # print(f"  Top 5 Probabilities: {', '.join(top_5_probs)}")
            # print()
            pbar.update(1)
            pbar.reset()
            i = i + 1
            sleep(0.1)
