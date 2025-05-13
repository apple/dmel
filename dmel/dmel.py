#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

from typing import Any, Dict, Tuple
import librosa
import os

import numpy as np
import torch
from einops import rearrange, repeat

from .utils import add_bos_eos, num_sample_per_frame

HAMMING = "HAMMING"
HANNING = "HANNING"

WINDOW_TYPE = {HAMMING: torch.hamming_window, HANNING: torch.hann_window}


class PowerSpectrum(torch.nn.Module):
    """
    Power spectrum computation from the raw wave, parameters follow torch.stft
    """

    def __init__(
        self,
        win_length: int,
        hop_length: int,
        n_fft: int,
        window_type: str = HANNING,
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
    ):
        super().__init__()
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.center = center
        self.normalized = normalized
        self.onesided = onesided
        if window_type not in WINDOW_TYPE:
            raise ValueError(f"{window_type} window is not implemented")
        self.window_type = window_type

    @torch.no_grad
    def forward(
        self, x: torch.Tensor, x_length: torch.LongTensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Power spectrum computation from the raw wave

        Args:
            x (torch.Tensor): input of shape (B, T)
            x_length (torch.LongTensor): sequences length of shape (B)
        Returns:
            out: output with power spectrum of shape (B, T, N)
        """
        out = torch.stft(
            x,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            center=self.center,
            window=WINDOW_TYPE[self.window_type](
                self.win_length, dtype=x.dtype, device=x.device
            ),
            normalized=self.normalized,
            onesided=self.onesided,
            return_complex=True,
        )
        # TBD remove clamp?
        out = torch.clamp(out.real**2 + out.imag**2, min=1.0e-10) ** 0.5
        out = rearrange(out, "b n t -> b t n")
        out_length = None

        if x_length is not None:
            if self.center:
                # add padding happened in stft
                x_length = x_length + 2 * (self.n_fft // 2)

            out_length = (
                torch.div(x_length - self.n_fft, self.hop_length, rounding_mode="trunc")
                + 1
            )
        return out, out_length


class MelFbank(torch.nn.Module):
    """Compute mel filterbanks via librosa mel filters

    Args:
        n_filterbank (int): number of mel
        n_fft (int): number of mel
        sampling_freq (int): sampling rate in Hz
        low_freq (int): lowest frequence in Hz
        high_freq (int): highest frequence in Hz (if < 0 will use sampling_freq / 2)
        mel_floor (float): avoiding zero to apply log afterwards
        device (torch.device): compute device
    """

    def __init__(
        self,
        n_filterbank: int = 80,
        n_fft: int = 512,
        sampling_freq: int = 16000,
        low_freq: int = 0,
        high_freq: int = -1,
        mel_floor: float = 1e-10,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        if high_freq <= 0:
            high_freq = sampling_freq // 2

        self.mel_floor = torch.tensor(mel_floor).to(device)
        H = librosa.filters.mel(
            sr=sampling_freq,
            n_fft=n_fft,
            n_mels=n_filterbank,
            fmin=low_freq,
            fmax=high_freq,
        )
        self.register_buffer(
            "H", torch.from_numpy(rearrange(H, "c n -> n c")).float().to(device)
        )

    @torch.no_grad
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute mel-filterbanks

        Args:
            x (torch.Tensor): shape(B, T, N)
            x_lengths (torch.LongTensor): shape(B)
        Returns:
            Tuple[torch.LongTensor, torch.LongTensor]: shape(B, T, C), shape(B) where C is number of filterbanks
        """
        return torch.maximum(x @ self.H, self.mel_floor)


class LogMelFbank(torch.nn.Module):
    """
    Log mel-filterbanks computation with implementation standard to TTS (not ASR)
    - apply window function (e.g. HANNING)
    - compute FFT
    - compute spectrum (abs)
    - compute mel fbanks (via librosa)
    - compute log
    - apply augmentation, e.g. SpecAugment
    """

    def __init__(
        self,
        sampling_freq: int = 16000,
        n_fft: int = 1024,
        frame_size_ms: int = 50,
        frame_stride_ms: int = 25,
        window_type: str = HANNING,
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
        n_filterbank: int = 80,
        low_freq: int = 80,
        high_freq: int = 7600,
        augmentation: torch.nn.Module = None,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()

        self.sampling_freq = sampling_freq
        self.n_filterbank = n_filterbank
        self.n_fft = n_fft
        self.frame_size_ms = frame_size_ms
        self.frame_stride_ms = frame_stride_ms
        self.window_type = window_type
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.augmentation = augmentation

        self.power_spectrum_func = PowerSpectrum(
            n_fft=n_fft,
            win_length=num_sample_per_frame(sampling_freq, frame_size_ms),
            hop_length=num_sample_per_frame(sampling_freq, frame_stride_ms),
            window_type=window_type,
            center=center,
            normalized=normalized,
            onesided=onesided,
        )

        self.mel_func = MelFbank(
            sampling_freq=sampling_freq,
            n_fft=n_fft,
            n_filterbank=n_filterbank,
            low_freq=low_freq,
            high_freq=high_freq,
            device=device,
        )

    def get_vocoder_parameters(self) -> Dict[str, Any]:
        """Return parameters required by Vocoder"""
        return dict(
            sampling_freq=self.sampling_freq,
            n_fft=self.n_fft,
            frame_size_ms=self.frame_size_ms,
            frame_stride_ms=self.frame_stride_ms,
            window=self.window_type,
            n_filterbank=self.n_filterbank,
            low_freq=self.low_freq,
            high_freq=self.high_freq,
        )

    @torch.no_grad
    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.LongTensor = None,
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """
        Compute log mel-filterbanks

        Args:
            x (torch.Tensor): shape(B, T)
            x_lengths (torch.LongTensor): shape(B)
        Returns:
            Tuple[torch.LongTensor, torch.LongTensor]: shape(B, T', C), shape(B)
        """
        out, out_lengths = self.power_spectrum_func(x, x_lengths)
        out = self.mel_func(out)
        # compute log mel with base 10 (different from ASR used exp)
        out = out.log10()
        if self.augmentation is not None:
            out = self.augmentation(out, out_lengths)
        return out, out_lengths


class DiscretizedLogMelFbank(torch.nn.Module):
    """
    dMel: takes the log mel-filterbanks and discretize them,
    additionally prepending <bos> and appending <eos> tokens
    """

    def __init__(
        self,
        logmelfbank: torch.nn.Module,
        n_bits: int,
        discretizer: str = None,
        quantize_min_value: float = -7,
        quantize_max_value: float = 2,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.logmelfbank = logmelfbank
        self.n_bits = n_bits
        self.discretizer = discretizer

        if discretizer is None:
            self.quantize_max_value = quantize_max_value
            self.quantize_min_value = quantize_min_value
            step_sizes = np.round(
                (self.quantize_max_value - self.quantize_min_value) / (2**n_bits - 1), 4
            )
            self.discretizing_matrix = repeat(
                np.array(
                    [self.quantize_min_value + i * step_sizes for i in range(2**n_bits)]
                ),
                "bins -> fbanks bins",
                fbanks=self.n_filterbank,
            )
            self.codebook_vocab_size = 2**n_bits
            assert (
                self.codebook_vocab_size == self.discretizing_matrix.shape[-1]
            ), "wrong codebook size"
        else:
            assert os.path.exists(discretizer), f"{discretizer} file does not exist!"
            self.discretizing_matrix = np.load(discretizer)
            self.codebook_vocab_size = self.discretizing_matrix.shape[-1]
            self.quantize_max_value = self.discretizing_matrix.max()
            self.quantize_min_value = self.discretizing_matrix.min()

        self.pad_value = self.quantize_max_value + 1
        self.pad_id = 0 + self.codebook_vocab_size
        self.bos_id = 1 + self.codebook_vocab_size
        self.eos_id = 2 + self.codebook_vocab_size
        self.special_tokens = [self.pad_id, self.bos_id, self.eos_id]
        self.vocab_size = self.codebook_vocab_size + len(self.special_tokens)

        for i, _ in enumerate([self.bos_id, self.eos_id]):
            self.discretizing_matrix = np.insert(
                self.discretizing_matrix,
                self.discretizing_matrix.shape[-1],
                self.quantize_max_value + i + 2,
                axis=-1,
            )
        self.discretizing_matrix = (
            torch.from_numpy(self.discretizing_matrix).float().to(device)
        )

    @property
    def n_filterbank(self) -> int:
        return self.logmelfbank.n_filterbank

    @torch.no_grad
    def inv_discretize_func(self, x: torch.LongTensor) -> torch.Tensor:
        """
        Tranform discritized mel index to discritized mel value

        Args:
            x (torch.LongTensor): shape of (B, T, C)
        Returns:
            torch.Tensor: shape of (B, T, C)
        """
        x_shape = x.shape
        if len(x_shape) == 2:
            x = rearrange(x, "t c -> 1 t c")
        melfbank = self.discretizing_matrix[
            np.arange(self.discretizing_matrix.shape[0]), x
        ]
        if len(x_shape) == 2:
            melfbank = rearrange(melfbank, "1 t c -> t c")
        return melfbank

    @torch.no_grad
    def discretize_func(self, x: torch.Tensor) -> torch.LongTensor:
        """Discretize the input tensor (mel feature).

        Args:
            x (torch.Tensor): shape (B, T, C)

        Returns:
            torch.LongTensor: shape (B, T, C)
        """
        x = torch.where(x > self.quantize_max_value, self.quantize_max_value, x)
        x = torch.where(
            (x < self.quantize_min_value) & (x != self.pad_value),
            self.quantize_min_value,
            x,
        )
        return (
            torch.abs(
                rearrange(x, "b t c -> b t c 1")
                - rearrange(self.discretizing_matrix, "c bins -> 1 1 c bins")
            )
            .argmin(-1)
            .long()
        )

    @torch.no_grad
    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.LongTensor = None,
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """
        Discretized Feature Extraction.

        Args:
            x (torch.Tensor): shape(B, T)
            x_lengths (torch.LongTensor): shape(B)
        Returns:
            Tuple[torch.LongTensor, torch.LongTensor]: shape(B, T', C), shape(B)
        """
        feature, feature_length = self.logmelfbank(x, x_lengths)
        discretized_feature = self.discretize_func(feature)
        discretized_feature_with_special_tokens, feature_length = add_bos_eos(
            discretized_feature, feature_length, self.bos_id, self.eos_id, self.pad_id
        )

        return discretized_feature_with_special_tokens, feature_length
