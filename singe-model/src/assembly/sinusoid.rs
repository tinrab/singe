//! Sinusoid features of a position, as `diffusers`' timestep embedding and the audio encoders' positions compute them.

use singe::{
    DataType, Device, Result, Shape, Tensor,
    graph::{DimensionExpr, capture},
};

/// How the frequencies are computed, as each reference computes them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SinusoidArithmetic {
    /// In `F32`, dividing by `half - shift` through its reciprocal, as `diffusers`' CUDA kernel does.
    Reciprocal { shift: f32 },
    /// In `F64` over `half - 1`, rounded to `F32` at the end.
    Double,
}

/// Which half of the features comes first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SinusoidOrder {
    SinesFirst,
    CosinesFirst,
}

/// Returns the `width` features of `position`: the sines and cosines of the position at frequencies falling geometrically from one toward 1/10000.
#[must_use]
pub fn sinusoids(
    position: f64,
    width: usize,
    order: SinusoidOrder,
    arithmetic: SinusoidArithmetic,
) -> Vec<f32> {
    let half = width / 2;
    let (sines, cosines): (Vec<f32>, Vec<f32>) = match arithmetic {
        SinusoidArithmetic::Reciprocal { shift } => {
            let log = -(10_000_f64.ln() as f32);
            let reciprocal = 1.0 / (half as f32 - shift);
            let position = position as f32;
            (0..half)
                .map(|index| position * (log * index as f32 * reciprocal).exp())
                .map(|angle| (angle.sin(), angle.cos()))
                .unzip()
        }
        SinusoidArithmetic::Double => {
            let increment = 10_000.0_f64.ln() / (half - 1) as f64;
            (0..half)
                .map(|index| position * (-increment * index as f64).exp())
                .map(|angle| (angle.sin() as f32, angle.cos() as f32))
                .unzip()
        }
    };
    match order {
        SinusoidOrder::SinesFirst => sines.into_iter().chain(cosines).collect(),
        SinusoidOrder::CosinesFirst => cosines.into_iter().chain(sines).collect(),
    }
}

/// Returns the sinusoids of the distances `reach` down to `-reach`, `[2 · reach + 1, width]`, each a sine and a cosine at each frequency `10000^(-2i / width)` in turn, in `F32`, as Transformer-XL's and `NeMo`'s relative positions compute them.
/// The distances are computed on `device` from `reach`, so a program with run-time axes holds as many as its rows need.
pub fn relative_sinusoids(device: Device, reach: DimensionExpr, width: i64) -> Result<Tensor> {
    let half = width / 2;
    let frequencies = capture::literal(device, |device| {
        let values = (0..half)
            .map(|index| 1.0 / 10_000_f32.powf((2 * index) as f32 / width as f32))
            .collect();
        Tensor::from_vec(device, Shape::create([1, half])?, values)
    })?;
    let count = reach.clone() * 2 + 1;
    let distances = Tensor::from_dimensions(device, &[reach])?
        .sub(&Tensor::range(device, 0_i64, count)?)?
        .cast(DataType::F32)?
        .unsqueeze(1)?;
    let angles = distances.mul(&frequencies)?;
    Tensor::stack(&[angles.sin()?, angles.cos()?], -1)?.flatten(1, 2)
}
