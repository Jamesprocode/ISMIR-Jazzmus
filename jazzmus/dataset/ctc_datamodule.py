from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from jazzmus.dataset.ctc_dataset import CTCDataset
from jazzmus.dataset.data_preprocessing import ctc_batch_preparation

import gin


@gin.configurable
class CTCDataModule(LightningDataModule):
    def __init__(  # noqa: PLR0913
        self,
        fold: int,
        batch_size: int = 16,
        num_workers: int = 16,
        width_reduction: int = 2,
        split_enc: bool = False,
        harm_proc: bool = False,
        path_to_splits: str = "data/splits",
    ):
        super().__init__()
        self.fold = fold
        self.train_split = []
        self.val_split = []
        self.test_split = []
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.width_reduction = width_reduction
        self.split_enc = split_enc
        self.harm_proc = harm_proc
        self.path_to_splits = path_to_splits

        self.load_splits(path_to_splits=self.path_to_splits)

    def load_splits(self, path_to_splits: str = "data/splits"):
        with open(f"{path_to_splits}/train_{self.fold}.dat") as f:
            self.train_split = f.readlines()

        with open(f"{path_to_splits}/val_{self.fold}.dat") as f:
            self.val_split = f.readlines()

        with open(f"{path_to_splits}/test_{self.fold}.dat") as f:
            self.test_split = f.readlines()

    def setup(self, stage: str):
        if stage == "fit":
            self.train_ds = CTCDataset(
                split_files=[self.train_split, self.val_split, self.test_split],
                split="train",
                width_reduction=self.width_reduction,
            )
            self.val_ds = CTCDataset(
                split_files=[self.train_split, self.val_split, self.test_split],
                split="val",
                width_reduction=self.width_reduction,
            )

        if stage == "test":
            self.test_ds = CTCDataset(
                split_files=[self.train_split, self.val_split, self.test_split],
                split="test",
                width_reduction=self.width_reduction,
            )

        if stage == "predict":
            self.predict_ds = CTCDataset(
                split_files=[self.train_split, self.val_split, self.test_split],
                split="predict",
                width_reduction=self.width_reduction,
                sample_files=self.sample_files,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=ctc_batch_preparation,
            persistent_workers=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_ds,
            batch_size=1,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def get_w2i_and_i2w(self):
        try:
            return self.train_ds.w2i, self.train_ds.i2w
        except AttributeError:
            return self.test_ds.w2i, self.test_ds.i2w

    def get_max_seq_len(self):
        try:
            return self.train_ds.max_seq_len
        except AttributeError:
            return self.test_ds.max_seq_len

    def get_max_img_len(self):
        try:
            return self.train_ds.max_img_len
        except AttributeError:
            return self.test_ds.max_img_len
