import os
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

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


def label_to_index(word):
    # Return the position of the word in labels
    return torch.tensor(CLASSES.index(word))


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
        return waveform, label_to_index(label)


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
