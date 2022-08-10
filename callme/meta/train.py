"""Trainer utility for Prototypical Networks https://arxiv.org/pdf/1703.05175.pdf"""
import logging
from learn2learn.algorithms.lightning import LightningPrototypicalNetworks

__all__ = ["ProtonetLightningTrainer"]

class ProtonetLightningTrainer(LightningPrototypicalNetworks):
    pass