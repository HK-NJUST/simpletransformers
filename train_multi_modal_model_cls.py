from simpletransformers.classification import MultiModalClassificationModel


train_args = { 
    "task_name": "train_test1",
    "output_dir": "outputs/",
    "overwrite_output_dir": True,
    "num_train_epochs": 3,

    # "use_early_stopping": True,
    # "early_stopping_metric": "mcc",
    # "n_gpu": 2,
    # "manual_seed": 4,
    # "use_multiprocessing": False,
    "train_batch_size": 8,
    "eval_batch_size": 8,
    "evaluate_during_training": True,
    "eval_epoch_period": 1,
    # "config": {
    #     "output_hidden_states": True
    # }
}

num_labels = 88
model = MultiModalClassificationModel(
    "bert", "bert-base-uncased", num_labels=num_labels, args=train_args
)

train_data = "/home/work/video_hk/data/video_tag/240919/labels"
model.train_model(train_data, eval_data=train_data) # , data_type_extension='json'