from torch import nn
from callme.meta.utils import prototypical_loss

__all__ = ["PrototypicalCosineLoss"]


class PrototypicalLoss(nn.Module):

    def forward(self, inputs, targets, n_support):
        return prototypical_loss(input=inputs, target=targets, n_support=n_support)
