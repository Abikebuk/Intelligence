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
