import MLM
import config
import Classification

if __name__ == "__main__":
    to_run = [
        # "mlm_train",
        # "mlm_inference",
        "classify"
    ]

    mlm_conf = config.MLM
    classifier_conf = config.classifier

    if "mlm_train" in to_run:
        MLM.train(
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
            dataset_range=10,  # if None, will use the whole dataset
            batch_size=8,
            num_epoch=4
        )
