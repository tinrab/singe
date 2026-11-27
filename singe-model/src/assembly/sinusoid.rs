//! Sinusoid features of a position, as `diffusers`' timestep embedding, the audio encoders' positions, and the DETR family's image positions compute them.

use std::f32::consts::PI;

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

/// How [`sine_positions`] places each point's coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SinePositions {
    /// What each coordinate, counted from one, has subtracted before the axis's length divides it: 0 for DETR, 0.5 for Deformable DETR.
    pub offset: f32,
    /// The base of the periods, 10000 for DETR.
    pub temperature: f32,
}

impl SinePositions {
    /// DETR's positions: coordinates from one, under a temperature of 10000.
    pub const DETR: Self = Self {
        offset: 0.0,
        temperature: 10_000.0,
    };
}

/// Returns the sine positions of each point of `map`, `[B, C, H, W]`, as `[1, H · W, features]`, as `DetrSinePositionEmbedding` and its relatives compute them with normalization.
///
/// Half the features encode the point's row and half its column, each from its coordinate counted from one, less the offset, and scaled to (0, 2π), over the temperature to the power of each feature pair's share.
/// They are computed from the map's height and width, so a program with run-time axes follows each image's.
pub fn sine_positions(map: &Tensor, features: i64, positions: SinePositions) -> Result<Tensor> {
    const SIDE_OFFSET: f32 = 1e-6;
    let device = map.device();
    let half = features / 2;
    // Each sine and the cosine after it share a period, `temperature^(2k / half)` for pair `k`.
    let periods = capture::literal(device, |device| {
        let values = (0..half / 2)
            .map(|pair| positions.temperature.powf((2 * pair) as f32 / half as f32))
            .collect();
        Tensor::from_vec(device, Shape::create([1, half / 2])?, values)
    })?;
    // `[extent, half]`: each coordinate along an axis, its sine and cosine at each period in turn.
    let encode = |extent: DimensionExpr| -> Result<Tensor> {
        let mut coordinates =
            Tensor::range(device, 1_i64, extent.clone() + 1)?.cast(DataType::F32)?;
        if positions.offset != 0.0 {
            coordinates = coordinates.sub_scalar(f64::from(positions.offset))?;
        }

        let normalizer = Tensor::from_dimensions(device, &[extent])?
            .cast(DataType::F32)?
            .add_scalar(f64::from(SIDE_OFFSET))?;
        let angles = coordinates
            .div(&normalizer)?
            .mul_scalar(f64::from(2.0 * PI))?
            .unsqueeze(1)?
            .div(&periods)?;
        Tensor::stack(&[angles.sin()?, angles.cos()?], -1)?.flatten(1, 2)
    };
    let (height, width) = (map.dimension(2)?, map.dimension(3)?);
    let grid = [height.clone(), width.clone(), half.into()];
    let rows = encode(height)?.unsqueeze(1)?.expand_with(&grid)?;
    let columns = encode(width)?.unsqueeze(0)?.expand_with(&grid)?;
    Tensor::concat(&[rows, columns], 2)?
        .flatten(0, 1)?
        .unsqueeze(0)
}
