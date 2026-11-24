//! A diffusion model's denoiser, as `diffusers`' unconditional `UNets` are: a noisy sample and its timestep to the model's prediction, the noise for the schedulers that read it so.

use std::fmt::Debug;

use singe::Tensor;

use crate::error::Result;

pub trait Denoiser: Debug {
    /// Returns the channels, height, and width of the samples it reads.
    fn sample_shape(&self) -> [i64; 3];

    /// Returns its prediction for `sample`, `[B, C, H, W]`, at `timestep`, in `F32`.
    fn predict(&self, sample: &Tensor, timestep: f32) -> Result<Tensor>;
}

/// A denoiser conditioned on a context, as Stable Diffusion's `UNets` are on their text encoder's states.
pub trait ConditionalDenoiser: Debug {
    /// Returns the channels of the latents it reads.
    fn in_channels(&self) -> i64;

    /// Returns its prediction for `sample`, `[B, C, H, W]`, at `timestep`, conditioned on `context`, `[B, T, D]`, in `F32`.
    fn predict(&self, sample: &Tensor, timestep: f32, context: &Tensor) -> Result<Tensor>;
}
