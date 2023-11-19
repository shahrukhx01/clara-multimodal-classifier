from pathlib import Path

import numpy as np
import pandas as pd
import torch, torchaudio
from torch.nn.utils.rnn import pad_sequence

from clara_mm_classifiers.utils.tokenizer import Tokenizer
from clara_mm_classifiers.utils.data_util import get_log_melspec


class VoxPopuliDataset(torch.utils.data.Dataset):
    # Simple class to load the desired folders inside ESC-50

    def __init__(
        self,
        path: Path = Path("data/ESC-50-master"),
        sample_rate: int = 16000,
        split_name: str = "train",
    ):
        # Load CSV & initialize all torchaudio.transforms:
        self.path = path
        self.csv = pd.read_csv(path / f"{split_name}_sample.csv")
        self.sample_rate = sample_rate
        self.split_name = split_name
 

    def __getitem__(self, index):        
        text = str(self.csv.iloc[index]["prediction"])
        label = self.csv.iloc[index]["label"]
        row = self.csv.iloc[index]
        wav, _ = torchaudio.load(self.path / "voicefiles" / row["voicefile"])
        wav = wav #[:self.sample_rate * 10]
        return wav, text, label


    def __len__(self):
        # Returns length
        return len(self.csv)

tokenizer = Tokenizer()

def collate_fn(batch):
    audios = [item[0] for item in batch]
    texts = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    texts = [torch.tensor(tokenizer.encode(text, lang="de")) for text in texts]

    mels = [get_log_melspec(np.array(a), 16000) for a in audios]
    mel_lengths = [mel.shape[0] for mel in mels]
    mel_lengths = torch.tensor(mel_lengths)
    
    text_lengths = [text.size(0) for text in texts]
    text_lengths = torch.tensor(text_lengths)
    mels = pad_sequence(mels).squeeze(-1).permute(1,2,0).contiguous()
    texts = pad_sequence(texts).T.contiguous()
    labels = torch.FloatTensor(labels)

    return labels, mels, texts, text_lengths, mel_lengths
