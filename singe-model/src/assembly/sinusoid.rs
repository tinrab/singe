//! Sinusoid features of a position, as `diffusers`' timestep embedding and the audio encoders' positions compute them.

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

/// Returns the sinusoids of the distances `length - 1` down to `-(length - 1)`, `[2 · length - 1, width]` row-major, each a sine and a cosine at each frequency `10000^(-2i / width)` in turn, in `F32`, as Transformer-XL's and `NeMo`'s relative positions compute them.
#[must_use]
pub fn relative_sinusoids(length: i64, width: usize) -> Vec<f32> {
    let frequencies = (0..width / 2)
        .map(|index| 1.0 / 10_000_f32.powf((2 * index) as f32 / width as f32))
        .collect::<Vec<_>>();
    (-(length - 1)..length)
        .rev()
        .flat_map(|distance| {
            let distance = distance as f32;
            frequencies.iter().flat_map(move |frequency| {
                let angle = distance * frequency;
                [angle.sin(), angle.cos()]
            })
        })
        .collect()
}
