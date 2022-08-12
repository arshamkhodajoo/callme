"""Input augmentation and transformation utility for 16k Hz sound tensors"""


import random

import numpy as np
import torch
import torch.nn.functional as F


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

    def __init__(self, time_shift_ms: int=400, sound_rate :int=16000):
        self.time_shift_ms= time_shift_ms
        self.rate = sound_rate

    def __call__(self, audio):
        time_shift = int((self.time_shift_ms * self.rate) / 1000)
        time_shift_amount = np.random.randint(-time_shift, time_shift)

        if time_shift_amount > 0:
            time_shift_padding = (time_shift_amount, 0)
            time_shift_offset = 0
        else:
            time_shift_padding = (0, -time_shift_amount)
            time_shift_offset = -time_shift_amount

        padded_foreground = F.pad(audio, time_shift_padding, 'constant', 0)
        sliced_foreground = torch.narrow(padded_foreground, 1, time_shift_offset, self.rate)
        return sliced_foreground

class ApplyBackgroundNoise:
    """apply and concat background noise to 16k wav audio"""

    def __init__(self, bg_dataset, sound_rate: int=16000, volume: float=0.60) -> None:
        self.bg_dataset = bg_dataset
        self.rate = sound_rate
        assert volume < 1, "volume too big"
        self.volume = volume

    def __call__(self, audio):
        background_sample = random.choice(self.bg_dataset)[0]
        assert len(background_sample) > self.rate, "background sample too short"
        background_offset = random.randint(0, len(background_sample) - self.rate)
        background_clip = background_sample[background_offset:(background_offset+self.rate)]
        background_clip = background_clip.view((1, self.rate))
        background_volume = random.uniform(0, self.volume)
        background_noise = background_clip * background_volume
        audio_with_noise = background_noise + audio
        noise_clamped = torch.clamp(audio_with_noise, -1.0, 1.0)
        return noise_clamped