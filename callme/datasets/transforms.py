"""Input augmentation and transformation utility for 16k Hz sound tensors"""


import random

import numpy as np
import torch


class ApplyFixAudioLength:
    """Either pads or truncates an audio into a fixed length."""

    def __init__(self, time=1):
        # time in seconds
        self.time = time

    def __call__(self, audio):
        audio = audio.numpy()[0]
        sample_rate = 16_000
        length = int(self.time * sample_rate)
        if length < len(audio):
            audio = audio[:length]
        elif length > len(audio):
            audio = np.pad(audio, (0, length - len(audio)), "constant")
        return torch.Tensor(audio).view((1, *audio.shape))


class ApplyTimeshift:
    """Shifts an audio randomly."""

    def __init__(self, max_shift_seconds=0.2):
        self.max_shift_seconds = max_shift_seconds

    def __call__(self, audio):

        sample_rate = 16_000
        max_shift = (sample_rate * self.max_shift_seconds)
        shift = random.randint(-max_shift, max_shift)
        a = -min(0, shift)
        b = max(0, shift)
        audio = np.pad(audio.numpy()[0], (a, b), "constant")
        out = torch.tensor(audio[:len(audio) - a] if a else audio[b:])
        return out.view((1, *out.size()))


class ApplyBackgroundNoise:
    """apply and concat background noise to 16k wav audio"""

    def __init__(self, bg_dataset) -> None:
        self.bg_dataset = bg_dataset

    def find_good_noise(self):
        """This is temporary solution to empty wav file bug, some noise files are too short to be applied"""
        noise = random.choice(self.bg_dataset)
        if noise.size(1) < 16_000:
            return self.find_good_noise()
        
        return noise

    def __call__(self, audio):
        noise = self.find_good_noise()
        rate = 16_000

        assert noise.size(1) > rate, "background audio too short to apply noise"

        if noise.size(1) > audio.size(1):
            equal_size_chunks = int(noise.size(1) / audio.size(1))
            random_chunk_select = random.randint(0, equal_size_chunks)
            # 0-16000 -> n=1 -> n-1 * 16000 : n * 16000
            slice_start = random_chunk_select - 1
            noise = noise[0][slice_start * rate: random_chunk_select * rate]
            noise = noise.view((1, *noise.size()))

        percentage = random.uniform(0, 0.1)
        return audio * (1 - percentage) + noise * percentage
