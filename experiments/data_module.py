from typing import Optional, Union

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler, SequentialSampler

from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer

HUGGINGFACE_TOKEN = "hf_XXX"


class TextsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        tokenizer: str,
        val_size: float = 0.1,
        test_size: Union[float, int] = 0.1,
        num_workers: int = 1,
        undersampling: Optional[float] = None,
    ):
        super().__init__()

        self.dataset: pd.DataFrame = None
        self.train_dataset: Optional[pd.DataFrame] = None
        self.val_dataset: Optional[pd.DataFrame] = None
        self.test_dataset: Optional[pd.DataFrame] = None

        self.val_size = val_size
        self.test_size = test_size
        self.batch_size = batch_size

        self.tokenizer = tokenizer
        self.num_workers = num_workers

        self.undersampling = undersampling

        self.dataset_encoder = LabelEncoder()

        self.split_done = False

    def set_tokenizer(self, tokenizer: str):
        self.tokenizer = tokenizer

    def prepare_data(self) -> None:
        hf_dataset = load_dataset("Brand24/mms", use_auth_token=HUGGINGFACE_TOKEN)

        self.dataset = hf_dataset["train"].to_pandas()

        print(self.dataset.shape)

        self.dataset.rename(
            columns={"label": "sentiment", "original_dataset": "dataset"}, inplace=True
        )
        self.dataset["dataset_label"] = self.dataset_encoder.fit_transform(
            self.dataset["dataset"]
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.split_done:
            self._new_split()
            self.split_done = True

    def _new_split(self):
        print("Removing (dataset, sentiment) pairs with too few examples...")
        too_few_examples = self.dataset[["dataset", "sentiment"]].value_counts()
        too_few_examples = too_few_examples[too_few_examples < 4].reset_index()

        drop_index = []

        for _, row in too_few_examples.iterrows():
            to_drop = self.dataset.loc[
                (self.dataset["sentiment"] == row["sentiment"])
                & (self.dataset["dataset"] == row["dataset"])
            ].index
            drop_index.extend(to_drop)

        dataset = self.dataset.drop(index=drop_index)
        print(f"Removed {self.dataset.shape[0] - dataset.shape[0]} samples")

        if self.undersampling is not None:
            print("Undersampling...")
            self.test_dataset, dataset = train_test_split(
                dataset, test_size=self.undersampling, stratify=dataset[["dataset"]]
            )
            dataset = dataset.reset_index()

            too_few_examples = dataset[["dataset", "sentiment"]].value_counts()
            too_few_examples = too_few_examples[too_few_examples < 4].reset_index()

            drop_index = []

            for _, row in too_few_examples.iterrows():
                to_drop = dataset.loc[
                    (dataset["sentiment"] == row["sentiment"])
                    & (dataset["dataset"] == row["dataset"])
                ].index
                drop_index.extend(to_drop)

            dataset = dataset.drop(index=drop_index)

            print("Train, val, test split...")
            train_val, _ = train_test_split(
                dataset,
                test_size=self.test_size,
                stratify=dataset[["dataset", "sentiment"]],
            )
            _, self.test_dataset = train_test_split(
                self.test_dataset,
                test_size=0.01,
                stratify=self.test_dataset[["dataset", "sentiment"]],
            )
        else:
            print("Train, val, test split...")
            train_val, self.test_dataset = train_test_split(
                dataset,
                test_size=self.test_size,
                stratify=dataset[["dataset", "sentiment"]],
            )

        self.train_dataset, self.val_dataset = train_test_split(
            train_val,
            test_size=self.val_size,
            stratify=train_val[["dataset", "sentiment"]],
        )

        print(f"Train: {self.train_dataset.shape[0]}")
        print(f"Val: {self.val_dataset.shape[0]}")
        print(f"Test: {self.test_dataset.shape[0]}")

    def _get_texts_dataloader(self, dataset: pd.DataFrame, shuffle=False):
        ds = FineTuningDataset(
            dataset["text"],
            torch.from_numpy(dataset["sentiment"].to_numpy()),
            torch.from_numpy(dataset["dataset_label"].to_numpy()),
            tokenizer=self.tokenizer,
        )
        if shuffle:
            sampler = BatchSampler(
                RandomSampler(ds), batch_size=self.batch_size, drop_last=False
            )
        else:
            sampler = BatchSampler(
                SequentialSampler(ds), batch_size=self.batch_size, drop_last=False
            )

        return DataLoader(
            ds, sampler=sampler, batch_size=None, num_workers=self.num_workers
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self._get_texts_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self._get_texts_dataloader(self.val_dataset)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self._get_texts_dataloader(self.test_dataset)


class FineTuningDataset(Dataset):
    def __init__(
        self,
        texts: pd.Series,
        *arrays: torch.Tensor,
        tokenizer: str = "bert-base-multilingual-cased",
    ) -> None:
        self.texts = texts
        self.arrays = arrays

        self.set_tokenizer(tokenizer)

    def set_tokenizer(self, tokenizer: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        self.max_len = self.tokenizer.model_max_length
        if self.max_len > 10000:
            self.max_len = 512

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        tokens = self.tokenizer(
            self.texts.iloc[index].to_list(),
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=self.max_len,
        )
        return tokens, *[x[index] for x in self.arrays]
