from __future__ import annotations
import abc
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import torchmetrics
import torchmetrics.functional as FM
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from sklearn.preprocessing import LabelEncoder
from typing import Callable, Optional, Union
from sklearn.metrics import classification_report
import numpy as np
from collections import defaultdict
import pandas as pd
from transformers import AutoModel


class TransformerClassifier(pl.LightningModule):
    def __init__(
        self,
        base_transformer: Union[str, None],
        embedding_size: int = 768,
        hidden_size: int = 32,
        num_classes: int = 3,
        learning_rate: float = 1e-5,
        metric_average: str = "macro",
        freeze_emb: bool = False,
        dropout: float = 0.0,
        ds_encoder: Optional[LabelEncoder] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        if base_transformer is not None:
            self.transformer = AutoModel.from_pretrained(base_transformer)

            if "mt5" in base_transformer:
                self.transformer = self.transformer.get_encoder()

            if freeze_emb:
                for param in self.transformer.parameters():
                    param.requires_grad = False
        else:
            self.transformer = None

        self.learning_rate = learning_rate

        self.num_classes = num_classes
        self.metric_average = metric_average

        self.text_logger = None
        self.train_accuracy = torchmetrics.Accuracy(
            num_classes=self.num_classes, average=self.metric_average
        )
        self.val_accuracy = torchmetrics.Accuracy(
            num_classes=self.num_classes, average=self.metric_average
        )
        self.val_precision = torchmetrics.Precision(
            num_classes=self.num_classes, average=self.metric_average
        )
        self.val_recall = torchmetrics.Recall(
            num_classes=self.num_classes, average=self.metric_average
        )
        self.val_fscore = torchmetrics.F1(
            num_classes=self.num_classes, average=self.metric_average
        )

        self.test_metrics: dict[str, Callable[..., Tensor]] = {
            "acc": FM.accuracy,
            "precision": FM.precision,
            "recall": FM.recall,
            "f1": FM.f1,
        }

        self.ds_encoder = ds_encoder

        self.test_metrics_prefix = ""

    def set_test_metrics_prefix(self, prefix: str):
        self.test_metrics_prefix = prefix

    def set_ds_encoder(self, ds_encoder: LabelEncoder):
        self.ds_encoder = ds_encoder

    @staticmethod
    def get_model_by_type(
        cls_type: str, constructor_params: dict
    ) -> TransformerClassifier:
        if cls_type == "Linear":
            return LinearTransformerClassifier(**constructor_params)
        elif cls_type == "BiLSTM":
            return BiLSTMTransformerClassifier(**constructor_params)
        else:
            raise ValueError("Not supported model type")

    @staticmethod
    def get_transformer_by_name(name: str) -> str:
        if name == "LaBSE":
            return "sentence-transformers/LaBSE"
        elif name == "DistilmBERT":
            return "distilbert-base-multilingual-cased"
        elif name == "MPNet":
            return "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        elif name == "mBERT":
            return "bert-base-multilingual-cased"
        elif name == "mUSE-dist":
            return "sentence-transformers/distiluse-base-multilingual-cased-v2"
        elif name == "XLM-R":
            return "xlm-roberta-base"
        elif name == "XLM-R-dist":
            return "sentence-transformers/paraphrase-xlm-r-multilingual-v1"
        elif name == "mT5":
            return "google/mt5-base"
        elif name == "miniLM":
            return "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        else:
            raise ValueError("Nor supported transformer name")

    @abc.abstractmethod
    def forward(self, tokens):
        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        tokens, y, *_ = batch
        logits = self(tokens)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=1)
        self.train_accuracy(preds, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_acc", self.train_accuracy, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        tokens, y, *_ = batch
        logits = self(tokens)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=1)
        self.val_accuracy(preds, y)
        self.val_precision(preds, y)
        self.val_recall(preds, y)
        self.val_fscore(preds, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", self.val_accuracy, on_step=False, on_epoch=True)
        self.log("val_precision", self.val_precision, on_step=False, on_epoch=True)
        self.log("val_recall", self.val_recall, on_step=False, on_epoch=True)
        self.log("val_f1", self.val_fscore, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        tokens, y, ds_labels, *_ = batch
        logits = self(tokens)
        preds = logits.argmax(dim=1)
        return preds, y, ds_labels, logits

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        preds, y, ds_labels, logits = zip(*outputs)
        preds = torch.cat(preds).cpu()
        y = torch.cat(y).cpu()
        ds_labels = torch.cat(ds_labels).cpu()
        logits = torch.cat(logits).cpu()

        self._log_test_metrics(preds, y, log_metric_per_class=True)
        self.select_text_loggger()
        self._log_classification_report(preds, y)
        self._log_predictions(preds, y, logits, ds_labels)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def select_text_loggger(self):
        try:
            logger_iterator = iter(self.logger)
        except TypeError:
            logger_iterator = iter([self.logger])
        self.text_logger = next(
            (
                logger
                for logger in logger_iterator
                if callable(getattr(logger.experiment, "log_text", None))
            ),
            None,
        )
        if self.text_logger is None:
            print("No logger capable of log_text available")

    def _log_classification_report(self, preds, y):
        if self.text_logger is not None:
            report = classification_report(y, preds)
            self.text_logger.experiment.log_text(
                self.text_logger.run_id,
                text=report,
                artifact_file=f"{self.test_metrics_prefix}test_report.txt",
            )

    def _log_test_metrics(
        self,
        preds: torch.Tensor,
        y: torch.Tensor,
        category_name: str = "",
        log_metric_per_class=False,
    ):
        if category_name != "":
            category_name = f"_{category_name}"

        for metric_name, metric in self.test_metrics.items():
            unique_classes, counts = y.unique(return_counts=True)
            unique_classes = unique_classes.tolist()
            counts = counts.tolist()
            counts_dict = defaultdict(lambda: 0, zip(unique_classes, counts))

            metric_value_per_class = metric(
                preds, y, average=None, num_classes=self.num_classes
            ).cpu()
            if log_metric_per_class:
                for i, value in enumerate(metric_value_per_class):
                    if i in unique_classes:
                        self.log(
                            f"{self.test_metrics_prefix}test_{metric_name}{category_name}_{i}",
                            value,
                        )

            if self.metric_average == "macro":
                weights = np.array(
                    [1 if x in counts_dict else 0 for x in range(self.num_classes)]
                )
                weight_sum = len(unique_classes)
            elif self.metric_average == "weighted":
                weights = np.array([counts_dict[x] for x in range(self.num_classes)])
                weight_sum = counts.sum()

            else:
                raise ValueError(f"Metric average: {self.metric_average} not supported")

            metric_value = (metric_value_per_class * weights).sum() / weight_sum
            self.log(
                f"{self.test_metrics_prefix}test_{metric_name}{category_name}",
                metric_value,
            )

    def _log_predictions(self, y_pred, y_true, logits, file_labels):
        if self.text_logger is not None:
            if self.ds_encoder is not None:
                file_labels = self.ds_encoder.inverse_transform(file_labels)
            metadata_df = pd.DataFrame({"dataset": file_labels})

            df = metadata_df.assign(y_pred=y_pred, y_true=y_true)
            logits = logits.numpy()
            logit_columns = [f"logit_{i}" for i in range(logits.shape[-1])]
            df[logit_columns] = logits
            tmp_file = f"{self.test_metrics_prefix}predictions.pkl.bz2"
            df.to_pickle(tmp_file)
            self.text_logger.experiment.log_artifact(self.text_logger.run_id, tmp_file)


class BiLSTMTransformerClassifier(TransformerClassifier):
    def __init__(
        self,
        base_transformer: str,
        embedding_size: int,
        hidden_size: int = 32,
        num_classes: int = 3,
        learning_rate: float = 0.00001,
        metric_average: str = "macro",
        freeze_emb: bool = False,
        dropout: float = 0.0,
        ds_encoder: Optional[LabelEncoder] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            base_transformer,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            num_classes=num_classes,
            learning_rate=learning_rate,
            metric_average=metric_average,
            freeze_emb=freeze_emb,
            dropout=dropout,
            ds_encoder=ds_encoder,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_size * 2, num_classes)
        )
        self.bilstm = nn.LSTM(
            embedding_size,
            hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, tokens):
        last_hidden = self.transformer(**tokens).last_hidden_state

        packed = pack_padded_sequence(
            last_hidden,
            lengths=tokens["attention_mask"].sum(dim=1).cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, (h_n, _) = self.bilstm(packed)
        out = torch.cat([h_n[-1], h_n[-2]], dim=1)

        return self.classifier(out)


class LinearTransformerClassifier(TransformerClassifier):
    """
    Transformer model with a linear classifier on top.
    NOTE: When training a model for production, make sure the APIs are compatible.
    """

    def __init__(
        self,
        base_transformer: str,
        embedding_size: int,
        hidden_size: int = 768,
        num_classes: int = 3,
        learning_rate: float = 0.00001,
        metric_average: str = "macro",
        freeze_emb: bool = False,
        dropout: float = 0.0,
        ds_encoder: Optional[LabelEncoder] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            base_transformer,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            num_classes=num_classes,
            learning_rate=learning_rate,
            metric_average=metric_average,
            freeze_emb=freeze_emb,
            dropout=dropout,
            ds_encoder=ds_encoder,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(embedding_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, tokens):
        cls_token = self.transformer(**tokens).last_hidden_state[:, 0, :]

        return self.classifier(cls_token)
