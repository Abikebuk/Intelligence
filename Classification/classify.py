import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification

def classify(model_id, dataset_id):
    # Define model name and labels
    model_name = "bert-base-uncased"
    labels = [
        "question",
        "answer",
        "statement",
        "request",
        "greeting",
        "threat",
        "acknowledgement",
        "clarification",
        "disagreement",
        "toxic",
        "insult",
        "positive",
        "negative",
        "neutral",
        "joke",
        "sarcasm",
        "anecdote",
        "compliment",
        "criticism",
        "apology",
        "suggestion",
        "warning",
        "invitation",
        "small_talk",
        "exclamation",
        "complaint",
        "encouragement",
        "empathy"
    ]

    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(labels))

    # Sample text data
    text = load_dataset(dataset_id, split="train").select(range(100))["text"]

    # Preprocess text
    encoded_data = tokenizer(text, padding="max_length", truncation=True)

    # Convert to tensors
    input_ids = torch.tensor(encoded_data['input_ids'])
    attention_mask = torch.tensor(encoded_data['attention_mask'])

    # Feed data to model and get logits
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    # Get predicted class labels (argmax)
    predictions = torch.argmax(logits, dim=-1)

    # Get probabilities for each class
    probabilities = torch.softmax(logits, dim=-1)

    # Print predictions
    for i, text_data in enumerate(text):
        print(f"Text: {text_data}")
        # Sort probabilities in descending order
        sorted_probs, sorted_indices = torch.sort(probabilities[i], descending=True)
        # Print only the top 5 probabilities in one line
        top_5_probs = [f"{labels[sorted_indices[j]]}: {sorted_probs[j].item():.4f}" for j in range(5)]
        print(f"  Top 5 Probabilities: {', '.join(top_5_probs)}")
        print()