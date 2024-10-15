import time
import torch
from simpletransformers.classification import MultiModalClassificationModel


train_args = { 
    "task_name": "test_all",
    "pretrain_params": "/home/work/video_hk/projects/24q3/simpletransformers/outputs/train_03_multilingual_weightedfea_bert_res50_cls88_add_lr_240929/checkpoint-5825-epoch-5/pytorch_model.bin",
    "output_dir": "/home/work/video_hk/projects/24q3/simpletransformers/outputs/",
    "pretrain": True,
    "freeze_nlp": False,
    "freeze_cv": True,
    "overwrite_output_dir": True,
    "add_mi_layer": True,
    "feature_weight": True,
    "num_train_epochs": 10,
    "warmup_ratio": 0.2,
    "learning_rate": 1e-4,
    # "use_early_stopping": True,
    # "early_stopping_metric": "mcc",
    "n_gpu": 1,
    # "manual_seed": 4,
    # "use_multiprocessing": False,
    "save_steps": 0,
    "train_batch_size": 16,
    "eval_batch_size": 16,
    "evaluate_during_training": True,
    "eval_epoch_period": 0,
    # "config": {
    #     "output_hidden_states": True
    # }
}
# time.sleep(50000000000000000000)
num_labels = 88
# bert-base-multilingual-uncased, bert-base-uncased
model = MultiModalClassificationModel(
    "bert", "bert-base-multilingual-uncased", num_labels=num_labels, args=train_args
)

if train_args["pretrain"]:
    model.model.load_state_dict(torch.load(train_args["pretrain_params"]), strict=False)

# 指定labels未知，支持多个位置的数据
# train_data = ["/home/work/datasets/train03/train_240820_240901/labels", "/home/work/datasets/train02/train_240810_240820/labels", "/home/work/datasets/train01/train_240801_240810/labels"] 
# # "/home/work/video_hk/data/video_tag/240919/labels" "/home/work/video_hk/data/video_tag/train_240820_240901/labels"
train_data = ["/home/work/video_hk/data/video_tag/test_240731/labels"]
test_data = ["/home/work/video_hk/data/video_tag/test_240731/labels"]
model.train_model(train_data, eval_data=test_data) # , data_type_extension='json'