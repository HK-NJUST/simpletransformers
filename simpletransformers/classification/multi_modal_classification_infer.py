#!/usr/bin/env python
# coding: utf-8


from __future__ import absolute_import, division, print_function

import json
import logging
import math
import os
import random
import warnings
from dataclasses import asdict
from datetime import datetime, timedelta
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import torch
from scipy.stats import mode, pearsonr
from sklearn.metrics import (
    confusion_matrix,
    label_ranking_average_precision_score,
    matthews_corrcoef,
    mean_squared_error,
    classification_report
)
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm, trange
from transformers.optimization import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from torch.optim import AdamW
from transformers.optimization import Adafactor
from transformers import (
    WEIGHTS_NAME,
    BertConfig,
    BertModel,
    BertTokenizer,
)
from transformers.models.deprecated.mmbt.configuration_mmbt import MMBTConfig

from simpletransformers.classification.classification_utils import (
    ImageEncoder,
    InputExample,
    JsonlDataset,
    collate_fn,
    convert_examples_to_features,
    get_image_transforms,
)
from simpletransformers.classification.transformer_models.mmbt_model import (
    MMBTForClassification,
)
from simpletransformers.config.global_args import global_args
from simpletransformers.config.model_args import MultiModalClassificationArgs
from simpletransformers.config.utils import sweep_config_to_sweep_values

try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False

rank = os.getenv('RANK', 'default_value')
logger = logging.getLogger("video_tag_logger")
logger.setLevel(logging.INFO)


class MultiModalClassificationModelInfer:
    def __init__(
        self,
        model_type,
        model_name,
        multi_label=False,
        label_list=None,
        num_labels=None,
        pos_weight=None,
        args=None,
        use_cuda=True,
        cuda_device=-1,
        **kwargs,
    ):
        """
        Initializes a MultiModalClassificationModelInfer model.

        Args:
            model_type: The type of model (bert, xlnet, xlm, roberta, distilbert, albert)
            model_name: Default Transformer model name or path to a directory containing Transformer model file (pytorch_nodel.bin).
            multi_label (optional): Set to True for multi label tasks.
            label_list (optional) : A list of all the labels (str) in the dataset.
            num_labels (optional): The number of labels or classes in the dataset.
            pos_weight (optional): A list of length num_labels containing the weights to assign to each label for loss calculation.
            args (optional): Default args will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.
            use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
            cuda_device (optional): Specific GPU that should be used. Will use the first available GPU by default.
            **kwargs (optional): For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied.
        """  # noqa: ignore flake8"

        MODEL_CLASSES = {
            "bert": (BertConfig, BertModel, BertTokenizer),
        }

        self.args = self._load_model_args(model_name)

        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, MultiModalClassificationArgs):
            self.args = args
        
        if self.args.task_name:
            self.args.output_dir = os.path.join(self.args.output_dir, self.args.task_name)

        # 设置日志文件路径
        if not os.path.exists(self.args.output_dir):
            os.mkdir(self.args.output_dir)

        if "sweep_config" in kwargs:
            self.is_sweeping = True
            sweep_config = kwargs.pop("sweep_config")
            sweep_values = sweep_config_to_sweep_values(sweep_config)
            self.args.update_from_dict(sweep_values)
        else:
            self.is_sweeping = False

        if self.args.manual_seed:
            random.seed(self.args.manual_seed)
            np.random.seed(self.args.manual_seed)
            torch.manual_seed(self.args.manual_seed)
            if self.args.n_gpu > 0:
                torch.cuda.manual_seed_all(self.args.manual_seed)

        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]

        self.label_list = label_list
        if self.label_list and not num_labels:
            num_labels = len(self.label_list)
        elif self.label_list and num_labels:
            if len(self.label_list) != num_labels:
                raise ValueError(
                    f"Mismatch in num_labels ({num_labels}) and length of label_list ({len(label_list)})"
                )

        if num_labels and not self.label_list:
            self.label_list = [str(i) for i in range(num_labels)]

        if num_labels:
            self.transformer_config = config_class.from_pretrained(
                model_name, num_labels=num_labels, **kwargs
            )
            self.num_labels = num_labels
        else:
            self.transformer_config = config_class.from_pretrained(model_name, **kwargs)
            self.num_labels = self.transformer_config.num_labels

        self.multi_label = multi_label

        if use_cuda:
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
            else:
                raise ValueError(
                    "'use_cuda' set to True when cuda is unavailable."
                    " Make sure CUDA is available or set use_cuda=False."
                )
        else:
            self.device = "cpu"

        self.transformer = model_class.from_pretrained(
            model_name, config=self.transformer_config, **kwargs
        )
        self.config = MMBTConfig(self.transformer_config, num_labels=self.num_labels)
        self.config.__dict__['use_return_dict'] = True
        self.config.__dict__['feature_weight'] = self.args.feature_weight
        self.config.__dict__['add_mi_layer'] = self.args.add_mi_layer
        self.results = {}

        self.img_encoder = ImageEncoder(self.args)
        self.model = MMBTForClassification(
            self.config, self.transformer, self.img_encoder
        )

        self.tokenizer = tokenizer_class.from_pretrained(
            model_name, do_lower_case=self.args.do_lower_case, **kwargs
        )

        if self.args.special_tokens_list:
            self.tokenizer.add_tokens(
                self.args.special_tokens_list, special_tokens=True
            )
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.args.model_name = model_name
        self.args.model_type = model_type

        if model_type in ["camembert", "xlmroberta"]:
            warnings.warn(
                f"use_multiprocessing automatically disabled as {model_type}"
                " fails when using multiprocessing for feature conversion."
            )
            self.args.use_multiprocessing = False

        if self.args.wandb_project and not wandb_available:
            warnings.warn(
                "wandb_project specified but wandb is not available. Wandb disabled."
            )
            self.args.wandb_project = None

        if multi_label:
            self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        elif self.num_labels == 1:
            self.criterion = torch.nn.MSELoss()
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

    def train_model(
        self,
        train_data,
        files_list=None,
        image_path=None,
        text_label=None,
        labels_label=None,
        images_label=None,
        image_type_extension=None,
        data_type_extension=None,
        auto_weights=False,
        output_dir=None,
        show_running_loss=True,
        args=None,
        eval_data=None,
        verbose=True,
        **kwargs,
    ):
        """
        Trains the model using 'train_data'

        Args:
            data: Path to data directory containing text files (JSON) and image files OR a Pandas DataFrame.
                If a DataFrame is given, it should contain the columns [text, labels, images]. When using a DataFrame,
                image_path MUST be specified. The image column of the DataFrame should contain the relative path from
                image_path to the image.
                E.g:
                    For an image file 1.jpeg located in "data/train/";
                        image_path = "data/train/"
                        images = "1.jpeg"
            files_list (optional): If given, only the files specified in this list will be taken from data directory.
                files_list can be a Python list or the path (str) to a JSON file containing a list of files.
            image_path (optional): Must be specified when using DataFrame as input. Path to the directory containing the
                images.
            text_label (optional): Column name to look for instead of the default "text"
            labels_label (optional): Column name to look for instead of the default "labels"
            images_label (optional): Column name to look for instead of the default "images"
            image_type_extension (optional): If given, this will be added to the end of each value in "images".
            data_type_extension (optional): If given, this will be added to the end of each value in "files_list".
            auto_weights (optional): If True, weights will be used to balance the classes. Only implemented for multi label tasks currently.
            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            show_running_loss (optional): Set to False to prevent running loss from being printed to console. Defaults to True.
            args (optional): Optional changes to the args dict of the model. Any changes made will persist for the model.
            eval_data (optional): A DataFrame against which evaluation will be performed when evaluate_during_training is enabled. Is required if evaluate_during_training is enabled.
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use). E.g. f1=sklearn.metrics.f1_score.
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.

        Returns:
            global_step: Number of global steps trained
            training_details: Average training loss if evaluate_during_training is False or full training progress scores if evaluate_during_training is True
        """  # noqa: ignore flake8"

        
        results


    def eval_model(
        self,
        eval_data,
        files_list=None,
        image_path=None,
        text_label=None,
        labels_label=None,
        images_label=None,
        image_type_extension=None,
        data_type_extension=None,
        output_dir=None,
        verbose=True,
        silent=False,
        **kwargs,
    ):
        """
        Evaluates the model on eval_data. Saves results to output_dir.

        Args:
            data: Path to data directory containing text files (JSON) and image files OR a Pandas DataFrame.
                If a DataFrame is given, it should contain the columns [text, labels, images]. When using a DataFrame,
                image_path MUST be specified. The image column of the DataFrame should contain the relative path from
                image_path to the image.
                E.g:
                    For an image file 1.jpeg located in "data/train/";
                        image_path = "data/train/"
                        images = "1.jpeg"
            files_list (optional): If given, only the files specified in this list will be taken from data directory.
                files_list can be a Python list or the path (str) to a JSON file containing a list of files.
            image_path (optional): Must be specified when using DataFrame as input. Path to the directory containing the
                images.
            text_label (optional): Column name to look for instead of the default "text"
            labels_label (optional): Column name to look for instead of the default "labels"
            images_label (optional): Column name to look for instead of the default "images"
            image_type_extension (optional): If given, this will be added to the end of each value in "images".
            data_type_extension (optional): If given, this will be added to the end of each value in "files_list".
            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            verbose: If verbose, results will be printed to the console on completion of evaluation.
            silent: If silent, tqdm progress bars will be hidden.
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use). E.g. f1=sklearn.metrics.f1_score.
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.

        Returns:
            result: Dictionary containing evaluation results. (Matthews correlation coefficient, tp, tn, fp, fn)
            model_outputs: List of model outputs for each row in eval_data
        """  # noqa: ignore flake8"

        if text_label:
            self.args.text_label = text_label

        if text_label:
            self.args.labels_label = labels_label

        if text_label:
            self.args.images_label = images_label

        if text_label:
            self.args.image_type_extension = image_type_extension

        if text_label:
            self.args.data_type_extension = data_type_extension

        if not output_dir:
            output_dir = self.args.output_dir

        self._move_model_to_device()

        # If data is a tuple,
        # this is for early stopping and first element is data_path and second element is files_list
        if isinstance(eval_data, tuple):
            data, files_list = eval_data

        eval_dataset = self.load_and_cache_examples(
            eval_data,
            files_list=files_list,
            image_path=image_path,
            text_label=self.args.text_label,
            labels_label=self.args.labels_label,
            images_label=self.args.images_label,
            image_type_extension=self.args.image_type_extension,
            data_type_extension=self.args.data_type_extension,
            verbose=verbose,
            silent=silent,
        )
        os.makedirs(output_dir, exist_ok=True)

        result, model_outputs = self.evaluate(
            eval_dataset, output_dir, verbose=verbose, silent=silent, **kwargs
        )

        self.results.update(result)

        if verbose:
            logger.info(self.results)

        return result, model_outputs

    def evaluate(
        self,
        eval_dataset,
        output_dir,
        prefix="",
        verbose=True,
        silent=False,
        **kwargs,
    ):
        """
        Evaluates the model on eval_df.

        Utility function to be used by the eval_model() method. Not intended to be used directly.
        """

        device = self.device
        model = self.model
        args = self.args
        multi_label = self.multi_label
        eval_output_dir = output_dir

        results = {}

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=args.eval_batch_size,
            collate_fn=collate_fn,
            num_workers=self.args.dataloader_num_workers,
        )

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        model.eval()

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        if args.fp16:
            from torch.cuda import amp

        all_imgs = []
        for batch in tqdm(
            eval_dataloader, disable=args.silent or silent, desc="Running Evaluation"
        ):
            image_pathes = batch[-1]
            all_imgs.extend(image_pathes)
            batch = batch[:-1]
            batch = tuple(t.to(device) for t in batch)
            labels = batch[5]
            with torch.no_grad():
                inputs = self._get_inputs_dict(batch)

                outputs = model(**inputs)
                logits = outputs[0]  # Different from default behaviour
                tmp_eval_loss = self.criterion(logits, labels)

                if self.args.n_gpu > 1:
                    tmp_eval_loss = tmp_eval_loss.mean()
                eval_loss += tmp_eval_loss.item()

            nb_eval_steps += 1

            if preds is None:
                preds = torch.softmax(logits, dim=1).detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
            else:
                preds = np.append(
                    preds, torch.softmax(logits, dim=1).detach().cpu().numpy(), axis=0
                )
                out_label_ids = np.append(
                    out_label_ids, labels.detach().cpu().numpy(), axis=0
                )

        eval_loss = eval_loss / nb_eval_steps
        model_outputs = preds

        if args.regression is True:
            preds = np.squeeze(preds)
            model_outputs = preds

        model_outputs = preds
        if multi_label:
            preds = (preds > 0.5).astype(int)
        else:
            preds_score = np.max(preds, axis=1)
            preds = np.argmax(preds, axis=1)
        all_outs = []
        for imgp, pscore, pid, gt in zip(all_imgs, preds_score, preds, out_label_ids):
            line = f"{imgp}\t{pid==gt}\t{pscore}\t{pid}\t{gt}\n"
            all_outs.append(line)
        
        output_eval_file = os.path.join(eval_output_dir, "debug_show.txt")
        with open(output_eval_file, "w") as writer:
            writer.writelines(all_outs)

        def inter_cal(preds_score, preds, out_label_ids, thres=0.5):
            over_thres_idx = np.where(preds_score > thres)[0]
            after_p, after_label = preds[over_thres_idx], out_label_ids[over_thres_idx]
            after_ratio = sum(after_p == after_label) / after_p.shape[0]
            return after_ratio, after_p, after_label
        for thre in np.arange(0.7, 0.99, 0.03):
            print(thre, inter_cal(preds_score, preds, out_label_ids, thre))


        result = self.compute_metrics(preds, out_label_ids, **kwargs)
        result["eval_loss"] = eval_loss
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "a+") as writer:
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))

        return results, model_outputs

    def load_and_cache_examples(
        self,
        data,
        files_list=None,
        image_path=None,
        text_label=None,
        labels_label=None,
        images_label=None,
        image_type_extension=None,
        data_type_extension=None,
        evaluate=False,
        no_cache=False,
        verbose=True,
        silent=False,
    ):
        """
        Converts a list of InputExample objects to a TensorDataset containing InputFeatures. Caches the InputFeatures.

        Args:
            data: Path to data directory containing text files (JSON) and image files OR a Pandas DataFrame.
                If a DataFrame is given, it should contain the columns [text, labels, images]. When using a DataFrame,
                image_path MUST be specified. The image column of the DataFrame should contain the relative path from
                image_path to the image.
                E.g:
                    For an image file 1.jpeg located in "data/train/";
                        image_path = "data/train/"
                        images = "1.jpeg"
            files_list (optional): If given, only the files specified in this list will be taken from data directory.
                files_list can be a Python list or the path (str) to a JSON file containing a list of files.
            image_path (optional): Must be specified when using DataFrame as input. Path to the directory containing the
                images.
            text_label (optional): Column name to look for instead of the default "text"
            labels_label (optional): Column name to look for instead of the default "labels"
            images_label (optional): Column name to look for instead of the default "images"
            image_type_extension (optional): If given, this will be added to the end of each value in "images".
            data_type_extension (optional): If given, this will be added to the end of each value in "files_list".

        Utility function for train() and eval() methods. Not intended to be used directly.
        """  # noqa: ignore flake8"

        tokenizer = self.tokenizer
        args = self.args
        if isinstance(data, list):
            pass
        elif not isinstance(data, str):
            if not image_path:
                raise ValueError(
                    "data is not a str and image_path is not given. image_path must be specified when input is a DF"
                )
            else:
                data = data.rename(
                    columns={
                        text_label: "text",
                        labels_label: "labels",
                        images_label: "images",
                    }
                )

        transforms = get_image_transforms()

        if self.label_list:
            labels = self.label_list
        else:
            labels = [str(i) for i in range(self.num_labels)]

        dataset = JsonlDataset(
            data,
            tokenizer,
            transforms,
            labels,
            args.max_seq_length - args.num_image_embeds - 2,
            files_list=files_list,
            image_path=image_path,
            text_label=text_label,
            labels_label=labels_label,
            images_label=images_label,
            image_type_extension=image_type_extension,
            data_type_extension=data_type_extension,
            multi_label=self.multi_label,
        )
        return dataset

    def compute_metrics(self, preds, labels, **kwargs):
        """
        Computes the evaluation metrics for the model predictions.

        Args:
            preds: Model predictions
            labels: Ground truth labels
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use). E.g. f1=sklearn.metrics.f1_score.
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.

        Returns:
            result: Dictionary containing evaluation results. (Matthews correlation coefficient, tp, tn, fp, fn)
            wrong: List of InputExample objects corresponding to each incorrect prediction by the model
        """  # noqa: ignore flake8"
        assert len(preds) == len(labels)

        multi_label = self.multi_label
        extra_metrics = {}
        for metric, func in kwargs.items():
            extra_metrics[metric] = func(labels, preds)

        if self.args.regression:
            return {**extra_metrics}

        if multi_label:
            label_ranking_score = label_ranking_average_precision_score(labels, preds)
            return {**{"LRAP": label_ranking_score}, **extra_metrics}

        mcc = matthews_corrcoef(labels, preds)

        if self.model.num_labels == 2:
            tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
            return {
                **{"mcc": mcc, "tp": tp, "tn": tn, "fp": fp, "fn": fn},
                **extra_metrics,
            }
        else:
            cm = confusion_matrix(labels, preds)
            def calculate_recall_precision_multiclass(cm):
                recall = np.diag(cm) / np.sum(cm, axis=1)
                precision = np.diag(cm) / np.sum(cm, axis=0)
                return recall, precision

            recall, precision = calculate_recall_precision_multiclass(cm)
            # logger.info(f"Recall for each class: {recall}")
            # logger.info(f"Precision for each class: {precision}")

            # 使用 classification_report 生成详细报告
            temp_outs = []
            gt_idx = sorted(set(preds.tolist()+labels.tolist()))
            for idx, pred, sub_recall in zip(gt_idx, precision, recall):
                line = f"{idx}\t{pred}\t{sub_recall}\n"
                temp_outs.append(line)
            with open("/home/work/video_hk/projects/24q3/simpletransformers/data/video_tag_stat.txt", 'w') as fo:
                fo.writelines(temp_outs)
            report = classification_report(labels, preds, output_dict=True, zero_division=0.0)
            logger.info(f"data num: {report['weighted avg']['support']}; acc: {report['accuracy']}; weighted avg: {report['weighted avg']}")
            return {**{"mcc": mcc, "data num": report['weighted avg']['support'],"acc": report['accuracy'], "precision": report['weighted avg']['precision'], "recall": report['weighted avg']['recall']}, **extra_metrics}

    def predict(self, to_predict, image_path, image_type_extension=None):
        """
        Performs predictions on a list of text.

        Args:
            to_predict: A python dictionary to be sent to the model for prediction.
                The dictionary should be of the form {"text": [<list of sentences>], "images": [<list of images>]}.
            image_path: Path to the directory containing the image/images.
            image_type_extension (optional): If given, this will be added to the end of each value in "images".

        Returns:
            preds: A python list of the predictions (0 or 1) for each text.
            model_outputs: A python list of the raw model outputs for each text.
        """

        device = self.device
        model = self.model
        args = self.args
        multi_label = self.multi_label

        self._move_model_to_device()

        to_predict.update({"labels": ["0" for i in range(len(to_predict["text"]))]})
        to_predict = pd.DataFrame.from_dict(to_predict)

        eval_dataset = self.load_and_cache_examples(
            to_predict,
            image_path=image_path,
            evaluate=True,
            image_type_extension=image_type_extension,
            no_cache=True,
        )

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=args.eval_batch_size,
            collate_fn=collate_fn,
            num_workers=self.args.dataloader_num_workers,
        )

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        if args.fp16:
            from torch.cuda import amp

        for batch in tqdm(
            eval_dataloader, disable=args.silent, desc="Running Prediction"
        ):
            batch = tuple(t.to(device) for t in batch)
            labels = batch[5]
            with torch.no_grad():
                inputs = self._get_inputs_dict(batch)
                if self.args.fp16:
                    with amp.autocast(True):
                        outputs = model(**inputs)
                        logits = outputs[0]  # Different from default behaviour
                else:
                    outputs = model(**inputs)
                    logits = outputs[0]  # Different from default behaviour
                tmp_eval_loss = self.criterion(logits, labels)

                if self.args.n_gpu > 1:
                    tmp_eval_loss = tmp_eval_loss.mean()
                eval_loss += tmp_eval_loss.item()

            nb_eval_steps += 1

            if preds is None:
                preds = torch.sigmoid(logits).detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
            else:
                preds = np.append(
                    preds, torch.sigmoid(logits).detach().cpu().numpy(), axis=0
                )
                out_label_ids = np.append(
                    out_label_ids, labels.detach().cpu().numpy(), axis=0
                )

        eval_loss = eval_loss / nb_eval_steps
        model_outputs = preds

        if multi_label:
            preds = (preds > 0.5).astype(int)
        else:
            preds = np.argmax(preds, axis=1)

        return preds, model_outputs

    def calculate_weights(self, train_dataset):
        label_frequences = train_dataset.get_label_frequencies()
        label_frequences = [
            label_frequences[label] if label_frequences[label] > 0 else 1
            for label in self.label_list
        ]
        label_weights = (
            torch.tensor(label_frequences, device=self.device, dtype=torch.float)
            / len(train_dataset)
        ) ** -1

        return label_weights

    def _threshold(self, x, threshold):
        if x >= threshold:
            return 1
        return 0

    def _move_model_to_device(self):
        self.model.to(self.device)

    def _get_inputs_dict(self, batch):
        inputs = {
            "input_ids": batch[0],
            "input_modal": batch[2],
            "attention_mask": batch[1],
            "modal_start_tokens": batch[3],
            "modal_end_tokens": batch[4],
        }

        return inputs

    def _get_last_metrics(self, metric_values):
        return {metric: values[-1] for metric, values in metric_values.items()}

    def _create_training_progress_scores(self, multi_label, **kwargs):
        extra_metrics = {key: [] for key in kwargs}

        if multi_label:
            training_progress_scores = {
                "global_step": [],
                "LRAP": [],
                "train_loss": [],
                "eval_loss": [],
                **extra_metrics,
            }
        else:
            if self.model.num_labels == 2:
                training_progress_scores = {
                    "global_step": [],
                    "tp": [],
                    "tn": [],
                    "fp": [],
                    "fn": [],
                    "mcc": [],
                    "train_loss": [],
                    "eval_loss": [],
                    **extra_metrics,
                }
            elif self.model.num_labels == 1:
                training_progress_scores = {
                    "global_step": [],
                    "train_loss": [],
                    "eval_loss": [],
                    **extra_metrics,
                }
            else:
                training_progress_scores = {
                    "global_step": [],
                    "mcc": [],
                    "data num": [],
                    "acc": [],
                    "precision": [],
                    "recall": [],
                    "train_loss": [],
                    "eval_loss": [],
                    **extra_metrics,
                }

        return training_progress_scores

    def save_model(self, output_dir, model=None, results=None):
        os.makedirs(output_dir, exist_ok=True)

        if model and not self.args.no_save:
            # Take care of distributed/parallel training
            model_to_save = model.module if hasattr(model, "module") else model
            torch.save(
                model_to_save.state_dict(),
                os.path.join(output_dir, "pytorch_model.bin"),
            )
            self.tokenizer.save_pretrained(output_dir)
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
            self.transformer_config.architectures = [model_to_save.__class__.__name__]
            self.transformer_config.save_pretrained(output_dir)
            self.save_model_args(output_dir)

        if results:
            output_eval_file = os.path.join(output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                for key in sorted(results.keys()):
                    writer.write("{} = {}\n".format(key, str(results[key])))

    def save_model_args(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.args.save(output_dir)

    def _load_model_args(self, input_dir):
        args = MultiModalClassificationArgs()
        args.load(input_dir)
        return args

    def get_named_parameters(self):
        return [n for n, p in self.model.named_parameters()]



