// Reshapes between feature maps `[B, C, H, W]`, their tokens `[B, H · W, C]`, and the windows of a channels-last map `[B, H, W, C]`, which Swin, Swin2SR, SegFormer, SAM, DETR, and Depth Anything read.

use singe::{Error as SingeError, Result as SingeResult, Tensor};

// A map `[B, C, H, W]` as tokens `[B, H · W, C]`.
pub fn to_tokens(map: &Tensor) -> SingeResult<Tensor> {
    map.flatten(2, 3)?.transpose(1, 2)
}

// Tokens `[B, H · W, C]` as a map `[B, C, H, W]`.
pub fn to_map(tokens: &Tensor, (height, width): (i64, i64)) -> SingeResult<Tensor> {
    let &[batch, _, channels] = tokens.shape().dimensions() else {
        return Err(SingeError::invalid_argument("tokens"));
    };
    tokens
        .transpose(1, 2)?
        .reshape([batch, channels, height, width])
}

pub fn spatial(map: &Tensor) -> SingeResult<(i64, i64)> {
    let &[_, _, height, width] = map.shape().dimensions() else {
        return Err(SingeError::invalid_argument("feature map"));
    };
    Ok((height, width))
}

// `[B, H, W, C]` as windows of `window × window` points, `[B · windows, window², C]`, row by row.
pub fn partition_windows(input: &Tensor, window: i64) -> SingeResult<Tensor> {
    let &[batch, height, width, channels] = input.shape().dimensions() else {
        return Err(SingeError::invalid_argument("window partition input"));
    };
    input
        .reshape([
            batch,
            height / window,
            window,
            width / window,
            window,
            channels,
        ])?
        .permute(&[0, 1, 3, 2, 4, 5])?
        .reshape([
            batch * (height / window) * (width / window),
            window * window,
            channels,
        ])
}

// Windows, `[B · windows, window², C]`, back into `[B, H, W, C]`.
pub fn merge_windows(
    windows: &Tensor,
    window: i64,
    (batch, height, width): (i64, i64, i64),
) -> SingeResult<Tensor> {
    let &[_, _, channels] = windows.shape().dimensions() else {
        return Err(SingeError::invalid_argument("windows"));
    };
    windows
        .reshape([
            batch,
            height / window,
            width / window,
            window,
            window,
            channels,
        ])?
        .permute(&[0, 1, 3, 2, 4, 5])?
        .reshape([batch, height, width, channels])
}
