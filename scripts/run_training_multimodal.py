import torch, torchaudio
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl
from torchmetrics import functional
import soundfile as sf

import pandas as pd
from pathlib import Path

import os

import logging
from clara_mm_classifiers.datasets.voxpopuli_dataset import VoxPopuliDataset, collate_fn
from clara_mm_classifiers.models.clara_multimodal_linear_probe import CLARAAudioMultimodalLinearProbe

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    path = Path("/mnt/d/clara-experiments/datasets/voxpopuli/")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "/mnt/d/clara-experiments/clara-medium.ckpt"
    model = CLARAAudioMultimodalLinearProbe(num_classes=1, clara_checkpoint_path=model_path, clara_map_location=device)

    train_data = VoxPopuliDataset(path=path)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=16
    )

    trainer = pl.Trainer(max_epochs=1, accelerator="cuda", precision=16)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=train_loader)
