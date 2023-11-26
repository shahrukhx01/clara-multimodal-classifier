from typing import Literal

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from clara_mm_classifiers.models.clara import PLCLARA
from clara_mm_classifiers.models.encoders.layer_modules import MLPLayers
from clara_mm_classifiers.utils.accuracy import accuracy
from clara_mm_classifiers.utils.modeling_utils import get_optimiser


class CLARAAudioMultimodalLinearProbe(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        clara_checkpoint_path: str,
        clara_map_location: str = "cuda",
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        learning_rate_patience: int = 10,
        LR_sheduler_T_max: int = 40,
        LR_sheduler_warmup_steps: int = 5,
        LR_sheduler_min_lr: float = 0.0,
        LR_sheduler_decay: float = 1.0,
        lr_interval: Literal["epoch", "step"] = "epoch",
        *args,
        **kwargs
    ) -> None:

        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.feature_extractor = PLCLARA.load_from_checkpoint(
            clara_checkpoint_path, map_location=clara_map_location
        )
        self.feature_extractor.freeze()

        self.classifier = MLPLayers(
            [2 * self.feature_extractor._hparams.output_dim, 512, 128, num_classes],
            dropout=dropout,
        )

    def forward(self, text: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:

        text_features = self.feature_extractor.encode_text(text)
        audio_features = self.feature_extractor.encode_audio(audio)

        # Projection
        text_features = self.feature_extractor.model.text_transform(text_features)
        audio_features = self.feature_extractor.model.audio_transform(audio_features)

        text_features = F.normalize(text_features, dim=-1)
        audio_features = F.normalize(audio_features, dim=-1)
        
        return self.classifier(torch.cat((audio_features, text_features), 1))

    def training_step(self, batch, batch_idx):
        _, loss, acc = self._shared_eval_step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _, loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics, prog_bar=True)

    def test_step(self, batch, batch_idx):
        _, loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics)

    def _shared_eval_step(self, batch, batch_idx):
        labels, mels, texts, _, _ = batch
        y_hat = self(texts, mels).squeeze()
        loss = F.cross_entropy(y_hat, labels)
        acc = accuracy(y_hat, labels)[0] / labels.size(0)

        return y_hat, loss, acc

    def configure_optimizers(self):
        return get_optimiser(self)
