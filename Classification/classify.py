import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from progress_table import ProgressTable


def classify(model_id, dataset_id, label_list, output_file, dataset_range=None, save_location=None, batch_size=64,
             num_epoch=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=len(label_list))
    model.to(device)

    dataset = load_dataset(dataset_id, split="train")
    if dataset_range is not None:
        dataset = dataset.select(range(dataset_range))

    best_predictions = [None] * len(dataset)
    best_confidences = [0] * len(dataset)
    model.eval()

    table = ProgressTable(
        pbar_embedded=False,
        pbar_style="angled alt red blue",
        interactive=2,
        pbar_show_progress=True,
        pbar_show_throughput=True,
        pbar_show_eta=True,
        default_column_width=8,
        default_header_color="bold",
    )
    table.add_column("Epoch", width=3)
    table.add_column("Batch", width=16)
    table.add_column("Predicted", width=10)
    table.add_column("c1", width=15)
    table.add_column("val1", width=6)
    table.add_column("c2", width=15)
    table.add_column("val2", width=6)
    table.add_column("c3", width=15)
    table.add_column("val3", width=6)


    with torch.no_grad():
        for epoch in table(range(num_epoch), show_throughput=False, show_eta=True):
            table["Epoch"] = epoch
            dataset = dataset.shuffle(seed=42 + epoch)

            num_batches = len(dataset) // batch_size
            for i, batch in table(enumerate(range(0, len(dataset), batch_size)), total=num_batches,
                                  description="train epoch"):
                batch_data = dataset[i * batch_size:(i + 1) * batch_size]
                texts = batch_data["text"]

                encoded_data = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
                input_ids = encoded_data['input_ids'].to(device)
                attention_mask = encoded_data['attention_mask'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                probabilities = torch.softmax(logits, dim=-1)

                for j in range(len(texts)):
                    text, pred, prob = texts[j], predictions[j], probabilities[j]
                    global_index = i * batch_size + j
                    predicted_label = label_list[pred.item()]
                    confidence = torch.max(prob).item()

                    if confidence > best_confidences[global_index]:
                        best_confidences[global_index] = confidence
                        best_predictions[global_index] = {
                            'Text': text,
                            'Predicted Label': predicted_label,
                            'Confidences': prob.tolist()
                        }

                top_5_indices = torch.argsort(prob, descending=True)[:3]
                table.update("Batch", f"{i + 1}/{num_batches}")
                table.update("Predicted", predicted_label)
                table.update("c1", label_list[0])
                table.update("c2", label_list[1])
                table.update("c3", label_list[2])
                table.update("val1", prob[0])
                table.update("val2", prob[1])
                table.update("val3", prob[2])
            table.next_row(split=epoch)

    table.close()

    # Write the best predictions to CSV (your existing code here)
    # ...

    print(f"Classification completed for {num_epoch} epochs. Best results saved to {output_file}")

# Example usage
# classify('bert-base-uncased', 'ag_news', ['World', 'Sports', 'Business', 'Sci/Tech'], 'output.csv')
