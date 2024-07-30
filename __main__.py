import json

import MLM
import config
import Classification
from Inference import run_inference
from PreTraining.Alpaca import pretrain_alpaca, parse_output
from Training import train
import torch.multiprocessing as mp

if __name__ == "__main__":
    to_run = [
        # "mlm_train",
        # "mlm_inference",
        # "classify",
        # "train",
        # "inference",
        "alpaca"
    ]

    mlm_conf = config.MLM
    classifier_conf = config.classifier
    trainer_conf = config.trainer

    if "mlm_train" in to_run:
        MLM.mlm_train(
            model_id=mlm_conf.model_id,
            dataset_id=mlm_conf.dataset_id,
            num_epochs=mlm_conf.num_epochs,
            batch_size=mlm_conf.batch_size,
            max_dataset_size=mlm_conf.max_dataset_size,
        )
    if "mlm_inference" in to_run:
        MLM.inference(
            model_id=mlm_conf.model_id,
            weight_location=mlm_conf.model_path,  # location is model_path (result from training)
            input_text="The paintbrush was angry at the color the artist chose to use.",
        )
    if "classify" in to_run:
        Classification.classify(
            model_id=classifier_conf.model_id,
            dataset_id=classifier_conf.dataset_id,
            label_list=classifier_conf.labels,
            dataset_range=None,  # if None, will use the whole dataset
            batch_size=8,
            num_epoch=2,
        )
    if "train" in to_run:
        train(
            model_id=trainer_conf.model_id,
            dataset_id=trainer_conf.dataset_id,
            batch_size=12,
            # label_filter=[
            #    "disgust",
            #    "annoyance",
            #    "excitement",
            #    "gratitude"
            #],
            gradient_accumulation_step=4,
            clear_cache_every_x_batches=1000,
            limit_dataset_size=10000,
            epochs=1,
        )
    if "inference" in to_run:
        run_inference(trainer_conf.model_id)

    if "alpaca" in to_run:
        mp.set_start_method('spawn')
        pretrain_alpaca()