"""Audio preprocessor utility used both in training and inference
"""
import torch
from torch import Tensor
from torchaudio.transforms import MFCC

__all__ = ["MFCCPreprocessor"]


class MFCCPreprocessor(MFCC):
    """convert 16k audio to  Mel Frequency Cepstral Coefficients (MFCCs)"""

    def __init__(self, window_size_ms: int = 40, window_stride_ms: int = 20, sample_rate: int = 16000, n_mfcc: int = 40, dct_type: int = 2, norm: str = "ortho", log_mels: bool = False) -> None:
        frame_len = window_size_ms / 1000
        stride = window_stride_ms / 1000

        super().__init__(sample_rate, n_mfcc, dct_type, norm, log_mels,
                         melkwargs={
                             'hop_length': int(stride*sample_rate),
                             'n_fft': int(frame_len*sample_rate)
                         })

    def forward(self, waveform: Tensor) -> Tensor:
        # single channel only
        mfcc_features = super().forward(waveform=waveform)[0]
        # f x t -> t x f
        mfcc_features = mfcc_features.T
        return torch.unsqueeze(mfcc_features, 0)
