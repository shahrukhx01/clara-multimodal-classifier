import librosa
import numpy as np
import torch

def get_log_melspec(audio, sr, n_mels=80, n_fft=1024, hop_length=512, win_length=1024, fmin=0, fmax=8000, **kwargs):
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, win_length=win_length, hop_length=hop_length, fmin=fmin, fmax=fmax, **kwargs)
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = (mel+40)/40
    return torch.tensor(mel, dtype=torch.float32).T
