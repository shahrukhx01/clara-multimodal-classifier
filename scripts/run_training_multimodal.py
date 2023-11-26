
from pathlib import Path

import logging

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from clara_mm_classifiers.datasets.meld_dataset import MELDDataset, collate_fn
from clara_mm_classifiers.models.clara_multimodal_linear_probe import CLARAAudioMultimodalLinearProbe

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # setup paths
    path = Path("/mnt/d/clara-experiments/datasets/meld")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load pre-trained base model weights and initialize classification layers
    model_path = "/mnt/d/clara-experiments/clara-medium.ckpt"
    model = CLARAAudioMultimodalLinearProbe(num_classes=7, clara_checkpoint_path=model_path, clara_map_location=device)

    # configure datasets and datloaders
    batch_size = 64
    num_dataloader_workers = 16
    train_data = MELDDataset(path=path)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_dataloader_workers
    )
    dev_data = MELDDataset(path=path, split_name="dev")
    dev_loader = torch.utils.data.DataLoader(
        dev_data, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_dataloader_workers
    )

    test_data = MELDDataset(path=path, split_name="test")
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_dataloader_workers
    )

    # configure trainer and start training
    mlf_logger = MLFlowLogger(experiment_name="clara-multimodal-classifier", tracking_uri="http://localhost:5000")
    early_stop_callback = EarlyStopping(monitor="val_acc", min_delta=0.00, patience=3, verbose=False, mode="max")

    trainer = pl.Trainer(max_epochs=20, logger=mlf_logger, accelerator="cuda", precision=16, callbacks=[early_stop_callback])
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=train_loader)

    # evaluate trained model
    trainer.test(model=model, dataloaders=test_loader)
