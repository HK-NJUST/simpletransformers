import time
from simpletransformers.classification import MultiModalClassificationModel


train_args = { 
    "task_name": "train_02_multilingual_bert_res50_cls88_add_lr_240926",
    "output_dir": "/home/work/video_hk/projects/24q3/simpletransformers/outputs/",
    "overwrite_output_dir": True,
    "num_train_epochs": 100,
    "warmup_ratio": 0.3,
    "learning_rate": 1e-4,
    # "use_early_stopping": True,
    # "early_stopping_metric": "mcc",
    # "n_gpu": 2,
    # "manual_seed": 4,
    # "use_multiprocessing": False,
    "save_steps": 0,
    "train_batch_size": 512,
    "eval_batch_size": 512,
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
# 指定labels未知，支持多个位置的数据
train_data = ["/home/work/datasets/train03/train_240820_240901/labels", "/home/work/datasets/train02/train_240810_240820/labels", "/home/work/datasets/train01/train_240801_240810/labels"] 
# # "/home/work/video_hk/data/video_tag/240919/labels" "/home/work/video_hk/data/video_tag/train_240820_240901/labels"
# train_data = ["/home/work/video_hk/data/video_tag/train_240820_240901/labels"]
test_data = ["/home/work/datasets/test/test_240731/labels"]
model.train_model(train_data, eval_data=test_data) # , data_type_extension='json'