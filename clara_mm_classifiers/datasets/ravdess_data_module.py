import pickle
from typing import Dict

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np

from clara_mm_classifiers.utils.tokenizer import Tokenizer
from clara_mm_classifiers.datasets.base_data_module import BaseDataModule
from clara_mm_classifiers.utils.data_util import get_log_melspec

OUTPUT_LABEL_KEY = "label"


class RavdessDataModule(BaseDataModule):
    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self.tokenizer = Tokenizer()

    def _read_pcikle(self) -> None:
        """Read the dataset from a pickle file.

        Args:
            dataset_path (Path): The path to the dataset.
        """
        with open(self.dataset_path, 'rb') as handle:
            self.dataset = pickle.load(handle)
      

    def setup(self, stage: str = "") -> None:
        """Setup the dataset for training and evaluation.

        Args:
            stage (str, optional): The stage of training. Defaults to "".
        """
        self._read_pcikle()
        for split in self.dataset:
            self.dataset[split] = self.dataset[split].map(self.convert_to_features, batched=True)
            self.columns = [column_name for column_name in self.dataset[split].column_names]
            self.dataset[split].set_format(type="torch", columns=self.columns)

    def prepare_data(self) -> None:
        """Download the dataset."""
        pass

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader.

        Returns:
            DataLoader: The training dataloader.
        """
        return DataLoader(
            self.dataset["train"],
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader.

        Returns:
            DataLoader: The validation dataloader.
        """
        return DataLoader(self.dataset["valid"], batch_size=self.eval_batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader.

        Returns:
            DataLoader: The test dataloader.
        """
        return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size, num_workers=self.num_workers)


    def convert_to_features(self, example_batch: dict[str, list[int | str]]) -> Dict:
        texts = [torch.tensor(self.tokenizer.encode(example)) for example in example_batch["text"]]
        label = torch.tensor([label for label in example_batch["label"]])
        
        mels = [get_log_melspec(np.array(a[0]), a[1]) for a in zip(example_batch['audio'], example_batch['sample_rate'])]
        del example_batch['audio']

        mel_lengths = [mel.shape[0] for mel in mels]
        mel_lengths = torch.tensor(mel_lengths)
        
        text_lengths = [text.size(0) for text in texts]
        text_lengths = torch.tensor(text_lengths)

        mels_padded = pad_sequence(mels).permute(1,2,0).contiguous()
        del mels
        # texts_padded = pad_sequence(texts).T.contiguous()

        return dict(label=label, mels=mels_padded, text_lengths=text_lengths, mel_lengths=mel_lengths)
