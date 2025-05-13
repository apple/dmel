#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

from typing import Tuple

import torch
from einops import rearrange, repeat


def num_sample_per_frame(sampling_rate_hz: int, frame_size_ms: int) -> int:
    return round(sampling_rate_hz * frame_size_ms / 1000.0)


def add_bos_eos(
    x: torch.LongTensor,
    x_lenght: torch.LongTensor,
    bos_id: int,
    eos_id: int,
    pad_id: int,
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """
    Add BOS and EOS tokens to the input tensor (discretized mel feature).

    Args:
        x (torch.LongTensor): shape(B, T, C)
        x_length (torch.LongTensor): shape(B)

    Returns:
        Tuple[torch.LongTensor, torch.LongTensor]: shape(B, T + 2, C), shape(B)
    """

    bos_token = x.new_full((x.shape[0], 1, x.shape[2]), bos_id)
    pad_token = x.new_full((x.shape[0], 1, x.shape[2]), pad_id)
    # Concatenate BOS tokens, input tensor, and EOS tokens
    input_with_bos_eos = torch.cat((bos_token, x, pad_token), dim=1).contiguous()
    x_lenght += 2

    batch_indices = rearrange(torch.arange(x.shape[0], device=x.device), "b -> b 1")
    eos_indices = rearrange((x_lenght - 1), "b -> b 1")
    input_with_bos_eos[batch_indices, eos_indices] = eos_id
    return input_with_bos_eos, x_lenght


def create_padding_mask(
    lengths: torch.LongTensor,
    max_T: int,
) -> torch.BoolTensor:
    """
    Create boolean padding mask (B, max_T) masking positions outside the lengths provided for every sequence.

    Args:
        lengths (torch.LongTensor): sequence length per sample in the batch, shape is (B).
        max_T (int): max sequence length - the output tensor shape

    Returns:
        torch.BoolTensor: boolean mask of the padded part (True at padded positions) of shape (B, max_T)
    """
    B = lengths.shape[0]
    mask = repeat(
        torch.arange(max_T, dtype=lengths.dtype, device=lengths.device), "t -> b t", b=B
    ) >= rearrange(lengths, "b -> b 1")
    return mask
