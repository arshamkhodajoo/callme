import os
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
import torchaudio
from torch import Tensor
from torchaudio.datasets import SPEECHCOMMANDS

__all__ = ["GoogleSpeechCommandDataset", "GoogleSpeechBackgroundDataset"]

CLASSES = ['backward',
           'bed',
           'bird',
           'cat',
           'dog',
           'down',
           'eight',
           'five',
           'follow',
           'forward',
           'four',
           'go',
           'happy',
           'house',
           'learn',
           'left',
           'marvin',
           'nine',
           'no',
           'off',
           'on',
           'one',
           'right',
           'seven',
           'sheila',
           'six',
           'stop',
           'three',
           'tree',
           'two',
           'up',
           'visual',
           'wow',
           'yes',
           'zero']


def file_is_background_audio(line):
    return "silence" in line or "unknown" in line


class GoogleSpeechCommandDataset(SPEECHCOMMANDS):
    """Wrapper for torchaudio's speech command dataset module"""

    def __init__(self, root: Union[str, Path], download: bool = False, subset: Optional[str] = None, transforms: Union[Callable, None] = None) -> None:
        super().__init__(root, download=download, subset=subset)

        assert subset in ["training", "validation",
                          "testing"], "subset not defined."

        if subset == "training":
            self._walker = self.load_list(
                "validation_list.txt") + self.load_list("testing_list.txt")
        if subset == "validation":
            self._walker = self.load_list("validation_list.txt")
        if subset == "testing":
            self._walker = self.load_list("testing_list.txt")

        self.transforms = transforms

    @staticmethod
    def labels_to_indices(word):
        # Return the position of the word in labels
        return torch.tensor(CLASSES.index(word))

    @staticmethod
    def indices_to_labels(index):
        # Return the word corresponding to the index in labels
        # This is the inverse of labels_to_indices
        return CLASSES[index]

    def get_label(self, fileid):
        """get indexed label from sample path"""
        relpath = os.path.relpath(fileid, self._path)
        label, _ = os.path.split(relpath)
        return self.labels_to_indices(label)

    @property
    def labels(self):
        return [self.get_label(fileid) for fileid in self._walker]

    def load_list(self, filename):
        """Read file and return list of sample filenames"""
        filepath = os.path.join(self._path, filename)
        with open(filepath) as fileobj:
            return [
                os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj
            ]

    def __getitem__(self, n: int) -> Tuple[Tensor, int]:
        waveform, sample_rate, label, speaker_id, utterance_number = super().__getitem__(n)
        if self.transforms is not None:
            waveform = self.transforms(waveform)
        return waveform, self.labels_to_indices(label)


class TripletSpeechCommandDataset(GoogleSpeechCommandDataset):
    """
    Triplet loss wrapper for GoogleSpeechCommand dataset
    returns (anchor, positive, negative)
    """

    def __init__(self, root: Union[str, Path], download: bool = False, subset: Optional[str] = None, transforms: Union[Callable, None] = None):
        super().__init__(root, download, subset, transforms)
        self.t_labels = torch.stack(self.labels)
        self.labels_set = set(self.t_labels.numpy())
        self.label_to_indices = {label: np.where(self.t_labels.numpy() == label)[0]
                                 for label in self.labels_set}

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        waveform, label = super().__getitem__(index)
        positive_index = index
        while positive_index == index:
            positive_index = np.random.choice(
                self.label_to_indices[int(label)])

        negative_label = np.random.choice(list(self.labels_set - set([label])))
        negative_index = np.random.choice(
            self.label_to_indices[int(negative_label)])
        positive_sample, _ = super().__getitem__(positive_index)
        negative_sample, _ = super().__getitem__(negative_index)

        return [(waveform, positive_sample, negative_sample), label]


class GoogleSpeechBackgroundDataset(GoogleSpeechCommandDataset):
    """Background noise and silence standalone dataset"""

    def __init__(self, root: Union[str, Path], download: bool = False, subset: Optional[str] = None) -> None:
        super().__init__(root, download, subset)
        self.background_noise_path = Path(
            "SpeechCommands/speech_commands_v0.02/_background_noise_")
        self.background_noise_path = Path(root) / self.background_noise_path

        assert self.background_noise_path.exists(
        ), "background speech commnad folder not found"
        self._walker = [
            file_path for file_path in self.background_noise_path.iterdir()
            if file_path.is_file() and ".wav" in str(file_path)
        ]

    def __getitem__(self, n: int) -> Tuple[Tensor, int]:
        # load only wav file and nothing else
        return torchaudio.load(self._walker[n])[0]
