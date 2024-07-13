import os
import pathlib
import pickle


def load_pickle(path):
    if os.path.exists(path):
        print(f"Loading pickle from {path}...")
        with open(path, "rb") as f:
            return pickle.load(f)
    else:
        print("File not found.")
        return None


def save_pickle(path, obj):
    create_dirs(path)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
        print(f"Saved pickle to {path}...")


def create_dirs(file_path: str):
    pathlib.Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def get_token_length_stats(dataset):
    print("Getting token stats for dataset...")
    dataset_token_length = [len(row['input_ids']) for row in dataset]
    dataset_token_length.sort()
    print(f"Dataset size: {len(dataset_token_length)}")
    print(f"Minimum: {dataset_token_length[0]} | Maximum: {dataset_token_length[-1]}")
    print("Average")
    print(f"Average: {sum(dataset_token_length) / len(dataset_token_length)}")
    print("Percentiles")
    print(
        f"5%: {dataset_token_length[int(len(dataset_token_length) * 0.05)]} | 15%: {dataset_token_length[int(len(dataset_token_length) * 0.15)]} | 25%: {dataset_token_length[int(len(dataset_token_length) * 0.25)]}")
    print(
        f"45%: {dataset_token_length[int(len(dataset_token_length) * 0.40)]} | 50%: {dataset_token_length[int(len(dataset_token_length) * 0.50)]} | 60%: {dataset_token_length[int(len(dataset_token_length) * 0.6)]}")
    print(
        f"75%: {dataset_token_length[int(len(dataset_token_length) * 0.05)]} | 85%: {dataset_token_length[int(len(dataset_token_length) * 0.85)]} | 95%: {dataset_token_length[int(len(dataset_token_length) * 0.95)]}")
    print(f"median: {statistics.median(dataset_token_length)}")
    print()
