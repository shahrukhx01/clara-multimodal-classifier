from typing import Literal, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from clara_mm_classifiers.loss.loss import CLARALoss
from clara_mm_classifiers.models.encoders.audio_encoder import *
from clara_mm_classifiers.models.encoders.layer_modules import (LayerNorm,
                                                                MLPLayers)
from clara_mm_classifiers.models.encoders.text_encoder import SimpleTransformer
from clara_mm_classifiers.utils.accuracy import Accuracy, accuracy
from clara_mm_classifiers.utils.modeling_utils import get_optimiser


class PerceiverIOEncoder(nn.Module):
    def __init__(
        self,
        *,
        num_layers,
        dim,
        num_latents=512,
        latent_dim=512,
        cross_heads=1,
        latent_heads=8,
        cross_dim_head=64,
        latent_dim_head=64,
        weight_tie_layers=False,
        seq_dropout_prob=0.0
    ):
        super().__init__()
        self.seq_dropout_prob = seq_dropout_prob

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        self.cross_attend_blocks = nn.ModuleList(
            [
                PreNorm(
                    latent_dim,
                    Attention(
                        latent_dim, dim, heads=cross_heads, dim_head=cross_dim_head
                    ),
                    context_dim=dim,
                ),
                PreNorm(latent_dim, FeedForward(latent_dim)),
            ]
        )

        get_latent_attn = lambda: PreNorm(
            latent_dim,
            Attention(latent_dim, heads=latent_heads, dim_head=latent_dim_head),
        )
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {"_cache": weight_tie_layers}

        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleList(
                    [get_latent_attn(**cache_args), get_latent_ff(**cache_args)]
                )
            )

    def forward(
        self,
        data,
        mask=None,
    ):
        b, *_, device = *data.shape, data.device
        data = data.permute(0, 2, 1)

        x = repeat(self.latents, "n d -> b n d", b=b)

        cross_attn, cross_ff = self.cross_attend_blocks

        # structured dropout (as done in perceiver AR https://arxiv.org/abs/2202.07765)

        if self.training and self.seq_dropout_prob > 0.0:
            data, mask = dropout_seq(data, mask, self.seq_dropout_prob)

        # cross attention only happens once for Perceiver IO

        x = cross_attn(x, context=data, mask=mask) + x
        x = cross_ff(x) + x

        # layers

        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        return x


class CLARA(nn.Module):
    """
    Contrastive Language-Speech Pre-training
    """

    def __init__(
        self,
        hparm,
        text_encoder: Optional[nn.Module] = None,
        audio_encoder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.hparm = hparm

        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder

        if self.text_encoder == None:
            self.text_encoder = SimpleTransformer(
                in_channels=self.hparm.text_encoder_embedding,
                out_channels=self.hparm.text_encoder_out_channels,
                num_layers=self.hparm.text_encoder_layers,
                nhead=self.hparm.text_encoder_heads,
                dropout=self.hparm.text_encoder_seq_dropout_prob,
                batch_first=True,
            )
            # self.text_encoder = PerceiverIOEncoder(
            # 	num_layers=self.hparm.text_encoder_layers,
            # 	dim=self.hparm.text_encoder_embedding,
            # 	num_latents=self.hparm.text_encoder_out_channels,
            # 	seq_dropout_prob=self.hparm.text_encoder_seq_dropout_prob,
            # 	)

        if self.audio_encoder == None:
            # self.audio_encoder = resnet18(1024)
            # self.audio_encoder = ResNeXt(5,12,1024, 2, 4)
            # self.audio_encoder = WhisperAudioEncoder(80, 1024, 1, 1)
            self.audio_encoder = PerceiverIOEncoder(
                num_layers=self.hparm.audio_encoder_layers,
                dim=self.hparm.audio_encoder_embedding,
                num_latents=self.hparm.audio_encoder_num_latents,
                latent_dim=self.hparm.audio_encoder_latent_dim,
                cross_heads=self.hparm.audio_encoder_cross_heads,
                latent_heads=self.hparm.audio_encoder_latent_heads,
                cross_dim_head=self.hparm.audio_encoder_cross_dim_head,
                latent_dim_head=self.hparm.audio_encoder_latent_dim_head,
                weight_tie_layers=self.hparm.audio_encoder_weight_tie_layers,
                seq_dropout_prob=self.hparm.audio_encoder_seq_dropout_prob,
            )

        # ------------
        # Text Layers
        # ------------
        self.text_embedding = nn.Embedding(
            self.hparm.vocab_size, self.hparm.text_encoder_embedding
        )
        self.text_positional_embedding = nn.Embedding(
            self.hparm.text_encoder_pos_embedding_size,
            self.hparm.text_encoder_embedding,
        )
        self.text_layer_norm = LayerNorm(self.hparm.text_encoder_out_channels)
        self.text_fc1 = nn.Linear(
            self.hparm.text_encoder_out_channels, self.hparm.text_encoder_project
        )
        self.text_transform = MLPLayers(
            units=[
                self.hparm.text_encoder_project,
                self.hparm.output_dim,
            ],
            dropout=self.hparm.text_encoder_project_dropout_prob,
        )

        # ------------
        # Audio Layers
        # ------------
        self.conv1 = nn.Conv1d(
            self.hparm.n_mels,
            self.hparm.audio_encoder_embedding,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = nn.Conv1d(
            self.hparm.audio_encoder_embedding,
            self.hparm.audio_encoder_embedding,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.audio_positional_embedding = nn.Embedding(
            self.hparm.audio_encoder_pos_embedding_size,
            self.hparm.audio_encoder_embedding,
        )
        self.audio_layer_norm = LayerNorm(self.hparm.audio_encoder_latent_dim)
        self.audio_fc1 = nn.Linear(
            self.hparm.audio_encoder_latent_dim, self.hparm.audio_encoder_project
        )
        self.audio_transform = MLPLayers(
            units=[
                self.hparm.audio_encoder_project,
                self.hparm.output_dim,
            ],
            dropout=self.hparm.audio_encoder_project_dropout_prob,
        )

        # ------------
        # Other
        # ------------
        self.audio_tempeture = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.text_tempeture = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_text(self, text: torch.Tensor):
        n, device = text.shape[1], text.device
        x = self.text_embedding(text)
        pos_emb = self.text_positional_embedding(torch.arange(n, device=device))
        pos_emb = rearrange(pos_emb, "n d -> () n d")
        x = x + pos_emb
        # x = x.permute(0,2,1) # (batch, seq, dim) -> (batch, dim, seq)
        x = self.text_encoder(x)
        # x = x.permute(0,2,1) # (batch, dim, seq) -> (batch, seq, dim)
        x = self.text_layer_norm(x)

        x1 = torch.mean(x, 1)
        x2, _ = torch.max(x, 1)
        x = x1 + x2

        x = F.leaky_relu(self.text_fc1(x))

        return x

    def encode_audio(self, audio: torch.Tensor):
        x = F.gelu(self.conv1(audio))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        pos_emb = self.audio_positional_embedding(
            torch.arange(x.shape[1], device=audio.device)
        )
        pos_emb = rearrange(pos_emb, "n d -> () n d")
        x = x + pos_emb
        x = x.permute(0, 2, 1)

        x = self.audio_encoder(x)
        x = self.audio_layer_norm(x)

        x1 = torch.mean(x, dim=2)
        x2, _ = torch.max(x, dim=2)
        x = x1 + x2

        x = F.leaky_relu(self.audio_fc1(x))

        return x

    def forward(self, text: torch.Tensor, audio: torch.Tensor):
        text_features = self.encode_text(text)
        audio_features = self.encode_audio(audio)

        # Projection
        text_features = self.text_transform(text_features)
        audio_features = self.audio_transform(audio_features)

        text_features = F.normalize(text_features, dim=-1)
        audio_features = F.normalize(audio_features, dim=-1)

        return (
            text_features,
            audio_features,
            self.text_tempeture.exp(),
            self.audio_tempeture.exp(),
        )


class PLCLARA(pl.LightningModule):
    def __init__(
        self,
        output_dim: int = 512,
        text_encoder_embedding: int = 1024,
        text_encoder_pos_embedding_size: int = 1024,
        text_encoder_width: int = 1024,
        text_encoder_layers: int = 1,
        text_encoder_heads: int = 4,
        text_encoder_out_channels: int = 512,
        text_encoder_project: int = 512,
        text_encoder_project_dropout_prob: float = 0.1,
        text_encoder_seq_dropout_prob: float = 0.5,
        vocab_size: int = 50373,
        n_mels: int = 80,
        audio_encoder_layers: int = 5,
        audio_encoder_embedding: int = 512,
        audio_encoder_pos_embedding_size: int = 4096,
        audio_encoder_num_latents: int = 512,
        audio_encoder_latent_dim: int = 512,
        audio_encoder_project: int = 512,
        audio_encoder_project_dropout_prob: float = 0.1,
        audio_encoder_cross_heads: int = 1,
        audio_encoder_latent_heads: int = 8,
        audio_encoder_cross_dim_head: int = 64,
        audio_encoder_latent_dim_head: int = 64,
        audio_encoder_weight_tie_layers: bool = False,
        audio_encoder_seq_dropout_prob: float = 0.5,
        learning_rate: float = 1e-3,
        learning_rate_patience: int = 10,
        LR_sheduler_T_max: int = 40,
        LR_sheduler_warmup_steps: int = 5,
        LR_sheduler_min_lr: float = 0.0,
        LR_sheduler_decay: float = 1.0,
        lr_interval: Literal["epoch", "step"] = "epoch",
    ):

        super().__init__()
        self.save_hyperparameters()

        self.model = CLARA(self.hparams)
        self.loss_fn = CLARALoss(cache_labels=True)
        self.acc_fn = Accuracy(cache_labels=True)

    def forward(self, texts: Optional[torch.Tensor], mels: Optional[torch.Tensor]):
        return self.model(texts, mels)

    def encode_audio(self, mels: torch.Tensor):
        return self.model.encode_audio(mels)

    def encode_text(self, text: torch.Tensor):
        return self.model.encode_text(text)

    def get_temps(self):
        return self.model.text_tempeture.exp(), self.model.audio_tempeture.exp()

    def training_step(self, batch, batch_idx):
        model_out, loss = self._shared_eval_step(batch, batch_idx)

        self.log("text_temp", model_out[2], sync_dist=True)
        self.log("audio_temp", model_out[3], sync_dist=True)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        _, loss, acc = self._shared_eval_step(batch, batch_idx)

        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics, prog_bar=True, sync_dist=True)
        return metrics

    def test_step(self, batch, batch_idx):
        _, loss, acc = self._shared_eval_step(batch, batch_idx)

        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics)

    def _shared_eval_step(self, batch, batch_idx):
        texts, mels, _, _ = batch  # torch.size([*, 123]), torch.size([*,80,1234])
        model_out = self(texts, mels)

        loss = self.loss_fn(*model_out)

        if self.training:
            return model_out, loss

        acc = self.acc_fn(*model_out)[0] / mels.size(0)
        return model_out, loss, acc

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        model_out, loss, acc = self._shared_eval_step(batch, batch_idx)
        return model_out, loss, acc

    def configure_optimizers(self):
        return get_optimiser(self)
