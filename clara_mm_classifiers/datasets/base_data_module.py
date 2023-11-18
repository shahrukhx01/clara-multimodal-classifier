from abc import ABC, abstractmethod
from typing import Dict

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


class BaseDataModule(LightningDataModule, ABC):
    def __init__(self, config: Dict):
        super().__init__()
        self.dataset_path = config["dataset_path"]
        self.num_workers = int(config.get("num_workers", 16))
        self.train_batch_size = int(config.get("train_batch_size", 32))
        self.eval_batch_size = int(config.get("eval_batch_size", 32))

    @abstractmethod
    def setup(self, stage: str = "") -> None:
        """Setup the dataset for training and evaluation.

        Args:
            stage (str, optional): The stage of training. Defaults to "".
        """
        raise NotImplementedError

    @abstractmethod
    def prepare_data(self) -> None:
        """Download the dataset."""
        raise NotImplementedError

    @abstractmethod
    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader.

        Returns:
            DataLoader: The training dataloader.
        """
        raise NotImplementedError

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader.

        Returns:
            DataLoader: The validation dataloader.
        """
        raise NotImplementedError

    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader.

        Returns:
            DataLoader: The test dataloader.
        """
        raise NotImplementedError
