//! Mamba-2's mixer as the hybrid decoders register it, Falcon-H1's beside attention and Granite-4.0-H's in place of it.

use singe::{
    Result as SingeResult,
    nn::{GatedNorm, Mamba2Layers, Mamba2Mixer, Mamba2Spec},
};

use crate::schema::Schema;

/// Which optional layers a Mamba-2 mixer has.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mamba2Layout {
    /// Whether the input and output projections have biases.
    pub projection_bias: bool,
    pub convolution_bias: bool,
    /// Whether the mixer norms its gated output, over groups of features.
    pub norm: bool,
}

/// Registers a mixer over `width` features: its `input`, `convolution`, `step_bias`, `decay_log`, `skip`, its gated `norm` if the layout has one, and `output`.
pub fn mamba2_mixer(
    schema: &mut Schema<'_>,
    width: i64,
    spec: Mamba2Spec,
    eps: f32,
    layout: Mamba2Layout,
) -> SingeResult<Mamba2Mixer> {
    let norm = layout
        .norm
        .then(|| -> SingeResult<GatedNorm> {
            Ok(GatedNorm {
                weight: schema.parameter("norm.weight", [spec.inner()])?,
                groups: spec.groups,
                eps,
            })
        })
        .transpose()?;
    let layers = Mamba2Layers {
        input: schema.linear("input", width, spec.projected(), layout.projection_bias)?,
        convolution: schema.parameter("convolution.weight", [spec.channels(), 1, spec.kernel])?,
        convolution_bias: layout
            .convolution_bias
            .then(|| schema.parameter("convolution.bias", [spec.channels()]))
            .transpose()?,
        step_bias: schema.parameter("step_bias", [spec.heads])?,
        decay_log: schema.parameter("decay_log", [spec.heads])?,
        skip: schema.parameter("skip", [spec.heads])?,
        norm,
        output: schema.linear("output", spec.inner(), width, layout.projection_bias)?,
    };
    Mamba2Mixer::create(layers, spec)
}
