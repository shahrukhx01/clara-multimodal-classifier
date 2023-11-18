import torch
import torch.nn as nn


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), -1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0,
                                                  keepdim=True).detach().cpu().numpy())
        for k in topk
    ]


class Accuracy(nn.Module):
    """
    CLAPLoss is adopted from the mlfoundations' open_clip: https://github.com/mlfoundations/open_clip
    """

    def __init__(self, top_k=(1,), cache_labels: bool = False) -> None:
        super().__init__()

        self.cache_labels = cache_labels
        self.top_k = top_k

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(
        self,
        text_features,
        audio_features,
        text_temperature: float = 1.0,
        audio_temperature: float = 1.0,
    ):
        device = audio_features.device

        logits_per_audio = audio_temperature * audio_features @ text_features.T
        logits_per_text = text_temperature * text_features @ audio_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_audio.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            # if self.world_size > 1 and self.local_loss:
            #     labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        acc = accuracy(logits_per_audio, labels, topk=self.top_k)
        return acc
