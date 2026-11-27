//! Reshapes between feature maps `[B, C, H, W]`, their tokens `[B, H · W, C]`, and the windows of a channels-last map `[B, H, W, C]`, which Swin, `Swin2SR`, `SegFormer`, SAM, DETR, and Depth Anything read.

use singe::{
    Error as SingeError, Result as SingeResult, Tensor,
    graph::DimensionExpr,
    shape::spatial::{Spatial2d, Spatial2dExpr, Spatial2dI64},
};

/// Returns a map `[B, C, H, W]` as tokens `[B, H · W, C]`.
pub fn to_tokens(map: &Tensor) -> SingeResult<Tensor> {
    map.flatten(2, 3)?.transpose(1, 2)
}

/// Returns tokens `[B, H · W, C]` as a map `[B, C, H, W]`, whose height and width may follow a program's run-time axes.
pub fn to_map(tokens: &Tensor, size: impl Into<Spatial2dExpr>) -> SingeResult<Tensor> {
    if tokens.shape().dimensions().len() != 3 {
        return Err(SingeError::invalid_argument("tokens"));
    }

    let Spatial2d { height, width } = size.into();
    let channels = tokens.transpose(1, 2)?;
    channels.reshape_with(&[
        channels.dimension(0)?,
        channels.dimension(1)?,
        height,
        width,
    ])
}

pub fn spatial(map: &Tensor) -> SingeResult<Spatial2dI64> {
    let &[_, _, height, width] = map.shape().dimensions() else {
        return Err(SingeError::invalid_argument("feature map"));
    };
    Ok(Spatial2d::new(height, width))
}

/// Returns a map's height and width as expressions, which follow a program's run-time axes.
pub fn spatial_dimensions(map: &Tensor) -> SingeResult<Spatial2dExpr> {
    spatial(map)?;
    Ok(Spatial2d::new(map.dimension(2)?, map.dimension(3)?))
}

/// Returns a map `[B, H, W, C]` as windows of `window × window` points, `[B · windows, window², C]`, row by row.
pub fn partition_windows(input: &Tensor, window: i64) -> SingeResult<Tensor> {
    let &[_, height, width, channels] = input.shape().dimensions() else {
        return Err(SingeError::invalid_argument("window partition input"));
    };
    let batch = input.dimension(0)?;
    input
        .reshape_with(&[
            batch.clone(),
            (height / window).into(),
            window.into(),
            (width / window).into(),
            window.into(),
            channels.into(),
        ])?
        .permute(&[0, 1, 3, 2, 4, 5])?
        .reshape_with(&[
            batch * ((height / window) * (width / window)),
            (window * window).into(),
            channels.into(),
        ])
}

/// Returns windows, `[B · windows, window², C]`, merged back into a map `[B, H, W, C]`.
pub fn merge_windows(
    windows: &Tensor,
    window: i64,
    batch: impl Into<DimensionExpr>,
    (height, width): (i64, i64),
) -> SingeResult<Tensor> {
    let &[_, _, channels] = windows.shape().dimensions() else {
        return Err(SingeError::invalid_argument("windows"));
    };
    let batch = batch.into();
    windows
        .reshape_with(&[
            batch.clone(),
            (height / window).into(),
            (width / window).into(),
            window.into(),
            window.into(),
            channels.into(),
        ])?
        .permute(&[0, 1, 3, 2, 4, 5])?
        .reshape_with(&[batch, height.into(), width.into(), channels.into()])
}
