from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder
import tempfile

import soundfile as sf


from clara_mm_classifiers.utils.tokenizer import Tokenizer
from clara_mm_classifiers.utils.data_util import get_log_melspec

label_encoder = LabelEncoder()

class MELDDataset(torch.utils.data.Dataset):
    # Simple class to load the desired folders inside ESC-50

    def __init__(
        self,
        path: Path = Path("data/ESC-50-master"),
        sample_rate: int = 16000,
        split_name: str = "train",
    ):
        # Load CSV & initialize all torchaudio.transforms:
        self.csv = pd.read_csv(path / f"{split_name}/{split_name}.csv")
        self.audio_files_path = path / f"{split_name}/voicefiles"
        self.sample_rate = sample_rate
        self.split_name = split_name
        # create transformed numerical label
        if split_name == "train":
            label_encoder.fit(self.csv.Emotion.values.tolist())
        self.csv["label"] = label_encoder.transform(self.csv.Emotion.values.tolist())

    def __getitem__(self, index): 
        current_row = self.csv.iloc[index]
        dialogue_id = current_row["Dialogue_ID"]
        utterance_id = current_row["Utterance_ID"]
        label = current_row["label"]

        file_path = f"{self.audio_files_path}/dia{dialogue_id}_utt{utterance_id}.mp3"
        
        audio, _ = sf.read(file_path)
        audio = audio.astype(float)
        audio = audio[:, 0]
        text = current_row["Utterance"]
        return audio, text, label


    def __len__(self):
        # Returns length
        return len(self.csv)

tokenizer = Tokenizer()

def collate_fn(batch):
    audios = [item[0] for item in batch]
    texts = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    texts = [torch.tensor(tokenizer.encode(text, lang="en")) for text in texts]

    mels = [get_log_melspec(np.array(a), 16000) for a in audios]
    mel_lengths = [mel.shape[0] for mel in mels]
    mel_lengths = torch.tensor(mel_lengths)
    
    text_lengths = [text.size(0) for text in texts]
    text_lengths = torch.tensor(text_lengths)

    mels = pad_sequence(mels).squeeze(-1).permute(1,2,0).contiguous()
    texts = pad_sequence(texts).T.contiguous()
    labels = torch.LongTensor(labels)

    return labels, mels, texts, text_lengths, mel_lengths
