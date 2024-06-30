from sklearn.utils import shuffle

class DatasetShuffler:
    def __init__(self, dataset_id: str, split: int = 0.2):
        self.dataset_id = dataset_id

    def shuffle_dataset(self):
        self.dataset_id = shuffle(data)