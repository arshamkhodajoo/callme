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

    def __init__(self, bg_dataset, sound_rate: int=16000, volume: float=0.80) -> None:
        self.bg_dataset = bg_dataset
        self.rate = sound_rate
        assert volume < 1, "volume too big"
        self.volume = volume

    def __call__(self, audio):
        background_sample = random.choice(self.bg_dataset)
        assert background_sample.size(1) > self.rate, "background sample too short"
        background_offset = random.randint(0, len(background_sample) - self.rate)
        background_clip = background_sample[background_offset:(background_offset+self.rate)]
        background_clip = background_clip.view((1, self.rate))
        background_volume = random.uniform(0, self.volume)
        background_noise = background_clip * background_volume
        audio_with_noise = background_noise + audio
        noise_clamped = torch.clamp(audio_with_noise, -1.0, 1.0)
        return noise_clamped
