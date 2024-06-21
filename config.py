class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


MLM = {
    "model_id": "distilbert/distilroberta-base",
    "dataset_id": "alpindale/light-novels",
    "num_epochs": 3,
    "batch_size": 20,
    "max_dataset_size": 10000,  # -1 for the whole dataset
    "tokens_path": "",  # default to tokens/mlm_dataset.pkl
    "model_path": ""  # default to models/mlm_model.pt
}
MLM = DotDict(MLM)

default = {
    "mlm_tokens_location": "tokens/mlm_dataset.pkl",
    "mlm_model_location": "models/mlm_model.pt",
}
default = DotDict(default)