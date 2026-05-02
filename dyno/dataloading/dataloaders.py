import os
import pandas as pd
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from .datasets import AudioDataset


def get_csv_annotations(csv_path: str, path_col: str = "npy_path") -> list[dict]:
    df = pd.read_csv(csv_path)
    annotations = df.to_dict("records")
    for annot in annotations:
        annot["file_path"] = annot[path_col]
    return annotations


def get_folder_annotations(data_path: str, extensions: tuple = (".mp3", ".wav", ".flac")) -> list[dict]:
    annotations = []
    for root, _, files in os.walk(data_path):
        for fname in sorted(files):
            if fname.lower().endswith(extensions):
                annotations.append({"file_path": os.path.join(root, fname)})
    return annotations


class FolderAudioDataModule(LightningDataModule):
    """Datamodule for raw audio extraction: scans a folder tree, no CSVs needed."""

    def __init__(
        self,
        folder_path: str,
        target_sr: int = 48000,
        target_n_samples: int = 96000,
        batch_size: int = 8,
        num_workers: int = 4,
        limit_n: int | None = None,
        extensions: list[str] | None = None,
    ):
        super().__init__()
        self.folder_path = folder_path
        self.target_sr = target_sr
        self.target_n_samples = target_n_samples
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.limit_n = limit_n
        self.extensions = tuple(extensions) if extensions else (".mp3", ".wav", ".flac")

    def setup(self, stage: str = None):
        self.train_dataset = AudioDataset(
            get_annotations_function=get_folder_annotations,
            task_kwargs={"data_path": self.folder_path, "extensions": self.extensions},
            preextracted_features=False,
            return_audio=True,
            target_sr=self.target_sr,
            target_n_samples=self.target_n_samples,
            limit_n=self.limit_n,
            split="train",
        )

    @property
    def val_datasets(self):
        return []

    @property
    def test_datasets(self):
        return []

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )


class AudioDataModule(LightningDataModule):
    def __init__(
        self,
        train_csv: str,
        val_csv: str,
        test_csv: str | None = None,
        path_col: str = "npy_path",
        n_frames: int = 60,
        batch_size: int = 32,
        num_workers: int = 4,
        limit_n: int | None = None,
        target_sr: int = 48000,
        target_n_samples: int = 96000,
        embedding_encoder: str = "unknown",
        embedding_rate: str = "unknown",
        embedding_dim: int | None = None,
    ):
        super().__init__()
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.path_col = path_col
        self.n_frames = n_frames
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.limit_n = limit_n
        self.target_sr = target_sr
        self.target_n_samples = target_n_samples
        self.embedding_encoder = embedding_encoder
        self.embedding_rate = embedding_rate
        self.embedding_dim = embedding_dim

    def _make_dataset(self, csv_path: str, split: str) -> AudioDataset:
        return AudioDataset(
            get_annotations_function=get_csv_annotations,
            task_kwargs={"csv_path": csv_path, "path_col": self.path_col},
            preextracted_features=True,
            n_frames=self.n_frames,
            split=split,
            limit_n=self.limit_n,
            target_sr=self.target_sr,
            target_n_samples=self.target_n_samples,
        )

    def setup(self, stage: str = None):
        self.train_dataset = self._make_dataset(self.train_csv, split="train")
        self.val_dataset = self._make_dataset(self.val_csv, split="val")
        if self.test_csv is not None:
            self.test_dataset = self._make_dataset(self.test_csv, split="test")

    # --- plural aliases used by gdr/extract_dataset.py ---
    @property
    def val_datasets(self):
        return [self.val_dataset] if hasattr(self, "val_dataset") else []

    @property
    def test_datasets(self):
        return [self.test_dataset] if hasattr(self, "test_dataset") else []

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )
