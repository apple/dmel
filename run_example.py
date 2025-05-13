#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import numpy as np
import torch
import torchaudio
from matplotlib import pyplot as plt
from typing import Optional, Tuple, List

import dmel


def plot_log_mel_spectrogram(
    data: List[np.ndarray],
    figsize: Tuple = (16, 4),
    span_boundary: Optional[List[List[int]]] = None,
    titles: Optional[List[str]] = None,
) -> plt.Figure:
    """Plotting mel spectrograms.

    Args:
        data (List): list of numpy array. The array size should be (Channels, Frames)
        figsize (tuple, optional): Defaults to (16, 4).
        span_boundary (List[List[int]], optional): Plot red lines for boundaries. Defaults to None.
        titles (List, optional): Titles of the figures. Defaults to None.

    Returns:
        matplotlib.Figure: melspectrogram figure
    """
    fig = plt.Figure(figsize=figsize)
    axes = fig.subplots(1, len(data))
    for i in range(len(data)):
        if len(data) == 1:
            axes.imshow(data[i], aspect="auto", origin="lower", interpolation="none")
        else:
            axes[i].imshow(data[i], aspect="auto", origin="lower", interpolation="none")
        if span_boundary:
            for x in span_boundary[i]:
                axes[i].axvline(x=x, color="red")
        if titles is not None:
            axes[i].title.set_text(titles[i])
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    # parameters setting
    sampling_freq = 16000
    frame_stride_ms = 12.5
    frame_size_ms = 50
    n_fft = 1024
    quantize_min_value = -7
    quantize_max_value = 2
    device = "cpu" if not torch.cuda.is_available() else torch.cuda.current_device()

    example_audio = torchaudio.load("example.wav")[0]
    example_audio_length = torch.LongTensor([example_audio.shape[1]])

    logmelfbank = dmel.LogMelFbank(
        sampling_freq=sampling_freq,
        n_fft=n_fft,
        frame_size_ms=frame_size_ms,
        frame_stride_ms=frame_stride_ms,
        n_filterbank=80,
    )

    # define dmel feature extraction
    dlogmelfbank = dmel.DiscretizedLogMelFbank(
        logmelfbank=logmelfbank,
        n_bits=4,
        quantize_min_value=quantize_min_value,
        quantize_max_value=quantize_max_value,
    )
    dmel_input, dmel_input_length = dlogmelfbank(example_audio, example_audio_length)
    # remove bos and eos frames
    dmel_input = dmel_input[:, 1:-1]
    dmel_input_length = dmel_input_length - 2
    print(dmel_input)
    # detokenize back
    mel_output = dlogmelfbank.inv_discretize_func(dmel_input)
    fig = plot_log_mel_spectrogram([mel_output[0].T])
    fig.savefig("example_dmel.png")

    mel_input, _ = logmelfbank(example_audio, example_audio_length)
    print(mel_input)
    fig = plot_log_mel_spectrogram([mel_input[0].T])
    fig.savefig("example_mel.png")
