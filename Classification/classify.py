import csv
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from progress_table import ProgressTable
# from sklearn.utils import shuffle
from config import default
import utils


def classify(model_id, dataset_id, label_list, output_file=default.classifier_output_location, dataset_range=None, batch_size=8,
             num_epoch=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=len(label_list))
    model.to(device)

    dataset = load_dataset(dataset_id, split="train").to_pandas().drop_duplicates(subset="text")
    if dataset_range is not None:
        dataset = dataset.head(dataset_range)

    best_predictions = [None] * len(dataset)
    best_confidences = [0] * len(dataset)
    model.eval()

    table = ProgressTable(
        pbar_embedded=False,
        pbar_style="angled alt red blue",
        pbar_show_progress=True,
        pbar_show_throughput=True,
        pbar_show_eta=True,
        default_header_color="bold",
    )
    table.add_column("Epoch", width=3)
    table.add_column("Batch", width=16)
    table.add_column("c0", width=15)
    table.add_column("val0", width=6)
    table.add_column("c1", width=15)
    table.add_column("val1", width=6)
    table.add_column("c2", width=15)
    table.add_column("val2", width=6)

    with torch.no_grad():
        for epoch in range(num_epoch):
            table["Epoch"] = epoch
            dataset = shuffle(dataset)
            epoch_result = []

            num_batches = len(dataset) // batch_size
            for i, batch in table(enumerate(range(0, len(dataset), batch_size)), total=num_batches):
                batch_data = dataset[i * batch_size:(i + 1) * batch_size]
                texts = batch_data["text"].to_list()

                encoded_data = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
                input_ids = encoded_data['input_ids'].to(device)
                attention_mask = encoded_data['attention_mask'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = outputs.loss
                predictions = torch.argmax(logits, dim=-1)
                probabilities = torch.softmax(logits, dim=-1)

                for j in range(len(texts)):
                    text, pred, prob = texts[j], predictions[j], probabilities[j]
                    predicted_label = label_list[pred.item()]

                    epoch_result.append({
                        'Text': text,
                        'Predicted Label': predicted_label,
                        'Confidences': prob.tolist()
                    })

                # Update the progress only with the last item of the batch (so it slightly prevent lag on terminal)
                # Pytorch terminal is slow.
                top_5_indices = torch.argsort(prob, descending=True)[:3]
                table.update("Batch", f"{i + 1}/{num_batches}")
                sorted_confidences = torch.argsort(prob, descending=True)
                for idx in range(3):
                    table.update(f"c{idx}", label_list[sorted_confidences[idx]])
                    table.update(f"val{idx}", prob[sorted_confidences[idx]])
            # Write results to CSV
            utils.create_dirs(output_file)
            with open(output_file, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                # Write the header row with label columns
                header = ['Text', 'Predicted Label'] + [f'{label}' for label in label_list]
                writer.writerow(header)
                for prediction in epoch_result:
                    if prediction is not None:
                        row = [prediction['Text'], prediction['Predicted Label']] + prediction['Confidences']
                        writer.writerow(row)
            table.next_row()
    table.close()
    print(f"Classification completed for {num_epoch} epochs. Best results saved to {output_file}")
