//! Padding that keeps a convolution's length or size, as `PyTorch`'s `padding="same"` and `kernel_size // 2` give it.

use singe::{
    nn::{Conv1dConfig, Conv2dConfig},
    shape::spatial::{Padding, Padding2d, Spatial2d},
};

#[must_use]
pub const fn same_1d(kernel: i64, dilation: i64) -> Conv1dConfig {
    let padding = dilation * (kernel - 1) / 2;
    Conv1dConfig {
        stride: 1,
        padding: Padding::symmetric(padding),
        dilation,
        groups: 1,
    }
}

#[must_use]
pub const fn same_2d(kernel: i64, stride: i64) -> Conv2dConfig {
    let padding = kernel / 2;
    Conv2dConfig {
        stride: Spatial2d::square(stride),
        padding: Padding2d::symmetric(padding, padding),
        dilation: Spatial2d::square(1),
        groups: 1,
    }
}
