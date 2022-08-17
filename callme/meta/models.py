""""
refernce:  https://arxiv.org/abs/1904.03814 and https://arxiv.org/abs/2007.14463
"""

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ["TCResNet8", "TCResNet", "TripletNetword"]


class Res8(nn.Module):
    """TCResnet8 from https://github.com/roman-vygon/triplet_loss_kws/blob/master/models/resnet.py"""

    def __init__(self, hidden_size):
        super().__init__()
        n_maps = hidden_size
        self.conv0 = nn.Conv2d(1, n_maps, (3, 3), padding=(1, 1), bias=False)
        self.pool = nn.AvgPool2d((3, 4))  # flipped -- better for 80 log-Mels

        self.n_layers = n_layers = 6
        self.convs = [nn.Conv2d(n_maps, n_maps, (3, 3),
                                padding=1, bias=False) for _ in range(n_layers)]
        for i, conv in enumerate(self.convs):
            self.add_module(f'bn{i + 1}', nn.BatchNorm2d(n_maps, affine=False))
            self.add_module(f'conv{i + 1}', conv)

    def forward(self, audio_signal):
        x = audio_signal.unsqueeze(1)
        # Original res8 uses (time, frequency) format
        x = x.permute(0, 1, 3, 2).contiguous()
        for i in range(self.n_layers + 1):
            y = F.relu(getattr(self, f'conv{i}')(x))
            if i == 0:
                if hasattr(self, 'pool'):
                    y = self.pool(y)
                old_x = y
            if i > 0 and i % 2 == 0:
                x = y + old_x
                old_x = x
            else:
                x = y
            if i > 0:
                x = getattr(self, f'bn{i}')(x)
        x = x.view(x.size(0), x.size(1), -1)  # shape: (batch, feats, o3)
        x = torch.mean(x, 2)
        return x.unsqueeze(-2)


class TripletNetwork(nn.Module):
    """Triplet wrapper for embedding model"""

    def __init__(self, embedding_model: Callable) -> None:
        super().__init__()
        self.embedding_model = embedding_model
        self.l2 = lambda output: output / \
            output.pow(2).sum(1, keepdim=True).sqrt()
        self.model = lambda x: self.l2(self.embedding_model(x))

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor):
        return (
            self.model(anchor),
            self.model(positive),
            self.model(negative)
        )
