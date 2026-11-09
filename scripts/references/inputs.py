"""Fixed inputs that Singe's tests compute the same way, so references need not store them."""

import math

import numpy
import torch


def pattern_image(size):
    """`[1, 3, size, size]` in [-1, 1]: integer patterns per channel, which `singe_testing::pattern_image` computes alike."""
    y = torch.arange(size).view(size, 1)
    x = torch.arange(size).view(1, size)
    channels = [
        (x * 3 + y) % 256,
        (x * y // 7 + 2 * y) % 256,
        ((x // 16 + y // 16) % 2) * 160 + (x + y) % 96,
    ]
    return (torch.stack(channels).to(torch.float32) / 127.5 - 1).unsqueeze(0)


def tone_and_chirp(samples):
    """A 440 Hz tone and a chirp rising from 300 Hz to 2 kHz at 16 kHz, in F64 and then F32, which `singe_testing::tone_and_chirp` computes alike."""
    values = []
    for index in range(samples):
        time = index / 16_000
        chirp = 300 + 1_700 * index / samples
        values.append(
            0.5 * math.sin(2 * math.pi * 440 * time)
            + 0.25 * math.sin(2 * math.pi * chirp * time)
        )
    return numpy.array(values, dtype=numpy.float32)
