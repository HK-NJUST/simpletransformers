import torch
from simpletransformers.classification import MultiModalClassificationModelInfer


train_args = { 
    "task_name": "test_240731_eval_0925_e40",
    "output_dir": "outputs/",
    "best_model_dir": "outputs/best_02",
    "overwrite_output_dir": True,
    "num_train_epochs": 30,

    # "use_early_stopping": True,
    # "early_stopping_metric": "mcc",
    # "n_gpu": 2,
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
params_path = "/home/work/video_hk/projects/24q3/simpletransformers/outputs/train_01_bert_res50_cls88_240925/checkpoint-93160-epoch-40/pytorch_model.bin"
num_labels = 88

model = MultiModalClassificationModelInfer(
    "bert", "bert-base-uncased", num_labels=num_labels, args=train_args
)
model.model.load_state_dict(torch.load(params_path))
# 指定labels未知，支持多个位置的数据
test_data = ["/home/work/video_hk/data/video_tag/test_240731/labels"]
results, _ = model.eval_model(test_data)
