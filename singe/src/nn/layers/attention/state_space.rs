use serde::{Deserialize, Serialize};

use crate::{
    data_type::DataType,
    error::{Error, Result},
    graph::{DimensionExpr, capture},
    nn::{AttentionInput, DecodingLayer, Linear, ModelState, StateCache, StateCacheSpec, StateKey},
    shape::Shape,
    tensor::{SequenceLayout, Tensor},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MambaSpec {
    pub inner: i64,
    pub state: i64,
    pub rank: i64,
    pub kernel: i64,
}

impl MambaSpec {
    #[must_use]
    pub const fn cache(&self) -> StateCacheSpec {
        StateCacheSpec {
            channels: self.inner,
            width: self.kernel,
            heads: self.inner,
            rows: 1,
            columns: self.state,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MambaLayers {
    pub input: Linear,
    pub convolution: StateKey,
    pub convolution_bias: Option<StateKey>,
    pub projection: Linear,
    pub step: Linear,
    pub decay_log: StateKey,
    pub skip: StateKey,
    pub output: Linear,
}

#[derive(Debug, Clone)]
pub struct MambaMixer {
    layers: MambaLayers,
    spec: MambaSpec,
}

impl MambaMixer {
    pub fn create(layers: MambaLayers, spec: MambaSpec) -> Result<Self> {
        if spec.inner <= 0 || spec.state <= 0 || spec.rank <= 0 || spec.kernel < 2 {
            return Err(Error::invalid_argument("mamba mixer sizes"));
        }
        Ok(Self { layers, spec })
    }
}

fn convolve(
    channels: &Tensor,
    taps: &StateKey,
    bias: Option<&StateKey>,
    cache: &StateCache,
    spans: SequenceLayout<'_>,
    state: &ModelState,
) -> Result<(Tensor, Tensor)> {
    let taps = state.tensor(taps)?;
    let &[width, _, kernel] = taps.shape().dimensions() else {
        return Err(Error::invalid_argument(
            "convolution taps are [channels, 1, kernel]",
        ));
    };
    let mut convolved =
        channels.causal_conv1d(&taps.reshape([width, kernel])?, cache.convolution(), spans)?;
    if let Some(bias) = bias {
        convolved = convolved.add(state.tensor(bias)?)?;
    }
    Ok((
        convolved,
        channels.causal_conv1d_state(cache.convolution(), spans)?,
    ))
}

impl DecodingLayer for MambaMixer {
    type Cache = StateCache;

    fn decode(&self, input: &AttentionInput<'_, StateCache>, state: &ModelState) -> Result<Tensor> {
        let (layers, spec) = (&self.layers, self.spec);
        let (hidden, cache, spans) = (input.hidden, input.cache, input.step.spans());
        let tokens = hidden.dimension(0)?;
        let mixed = hidden.apply(&layers.input, state)?;
        let channels = mixed.narrow(-1, 0, spec.inner)?.contiguous()?;
        let gate = mixed.narrow(-1, spec.inner, spec.inner)?;
        let (convolved, convolution) = convolve(
            &channels,
            &layers.convolution,
            layers.convolution_bias.as_ref(),
            cache,
            spans,
            state,
        )?;
        let convolved = convolved.silu()?;
        let projected = convolved.apply(&layers.projection, state)?;
        let projection = |start: i64| {
            projected
                .narrow(-1, start, spec.state)?
                .contiguous()?
                .reshape_with(&[tokens.clone(), 1.into(), spec.state.into()])
        };
        let (written, read) = (projection(spec.rank)?, projection(spec.rank + spec.state)?);
        let step = projected
            .narrow(-1, 0, spec.rank)?
            .apply(&layers.step, state)?
            .cast(DataType::F32)?
            .softplus(1.0)?;
        let rates = state
            .tensor(&layers.decay_log)?
            .cast(DataType::F32)?
            .exp()?
            .neg()?;
        let heads = convolved.reshape_with(&[tokens.clone(), spec.inner.into(), 1.into()])?;
        let scan = |spans: SequenceLayout<'_>| {
            heads.selective_scan(
                &step,
                &step,
                &rates,
                &written,
                &read,
                cache.recurrent(),
                spans,
            )
        };
        let (scanned, left) = scan(spans)?;
        cache.write_step(spans, &convolution, &left, |first| {
            Ok((
                channels.causal_conv1d_state(cache.convolution(), first)?,
                scan(first)?.1,
            ))
        })?;
        let skip = state.tensor(&layers.skip)?;
        scanned
            .reshape_with(&[tokens, spec.inner.into()])?
            .add(&convolved.mul(skip)?)?
            .mul(&gate.silu()?)?
            .apply(&layers.output, state)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Mamba2Spec {
    pub heads: i64,
    pub head_dim: i64,
    pub groups: i64,
    pub state: i64,
    pub kernel: i64,
}

impl Mamba2Spec {
    #[must_use]
    pub const fn inner(&self) -> i64 {
        self.heads * self.head_dim
    }

    #[must_use]
    pub const fn channels(&self) -> i64 {
        self.inner() + 2 * self.groups * self.state
    }

    #[must_use]
    pub const fn projected(&self) -> i64 {
        self.inner() + self.channels() + self.heads
    }

    #[must_use]
    pub const fn cache(&self) -> StateCacheSpec {
        StateCacheSpec {
            channels: self.channels(),
            width: self.kernel,
            heads: self.heads,
            rows: self.head_dim,
            columns: self.state,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GatedNorm {
    pub weight: StateKey,
    pub groups: i64,
    pub eps: f32,
}

#[derive(Debug, Clone)]
pub struct Mamba2Layers {
    pub input: Linear,
    pub convolution: StateKey,
    pub convolution_bias: Option<StateKey>,
    pub step_bias: StateKey,
    pub decay_log: StateKey,
    pub skip: StateKey,
    pub norm: Option<GatedNorm>,
    pub output: Linear,
}

#[derive(Debug, Clone)]
pub struct Mamba2Mixer {
    layers: Mamba2Layers,
    spec: Mamba2Spec,
}

impl Mamba2Mixer {
    pub fn create(layers: Mamba2Layers, spec: Mamba2Spec) -> Result<Self> {
        if spec.heads <= 0
            || spec.head_dim <= 0
            || spec.groups <= 0
            || spec.heads % spec.groups != 0
            || spec.state <= 0
            || spec.kernel < 2
        {
            return Err(Error::invalid_argument("mamba2 mixer sizes"));
        }

        if layers
            .norm
            .as_ref()
            .is_some_and(|norm| norm.groups <= 0 || spec.inner() % norm.groups != 0)
        {
            return Err(Error::invalid_argument("mamba2 norm groups"));
        }
        Ok(Self { layers, spec })
    }
}

impl DecodingLayer for Mamba2Mixer {
    type Cache = StateCache;

    fn decode(&self, input: &AttentionInput<'_, StateCache>, state: &ModelState) -> Result<Tensor> {
        let (layers, spec) = (&self.layers, self.spec);
        let (hidden, cache, spans) = (input.hidden, input.cache, input.step.spans());
        let data_type = hidden.data_type();
        let tokens = hidden.dimension(0)?;
        let (inner, width) = (spec.inner(), spec.groups * spec.state);
        let mixed = hidden.apply(&layers.input, state)?;
        let gate = mixed.narrow(-1, 0, inner)?;
        let channels = mixed.narrow(-1, inner, spec.channels())?.contiguous()?;
        let (convolved, convolution) = convolve(
            &channels,
            &layers.convolution,
            layers.convolution_bias.as_ref(),
            cache,
            spans,
            state,
        )?;
        let convolved = convolved.silu()?;
        let part = |start: i64, heads: i64, dim: i64| {
            convolved
                .narrow(-1, start, heads * dim)?
                .contiguous()?
                .reshape_with(&[tokens.clone(), heads.into(), dim.into()])
        };
        let features = part(0, spec.heads, spec.head_dim)?;
        let written = part(inner, spec.groups, spec.state)?;
        let read = part(inner + width, spec.groups, spec.state)?;
        let step = mixed
            .narrow(-1, inner + spec.channels(), spec.heads)?
            .cast(DataType::F32)?
            .add(&state.tensor(&layers.step_bias)?.cast(DataType::F32)?)?
            .softplus(1.0)?;
        let rates = state
            .tensor(&layers.decay_log)?
            .cast(DataType::F32)?
            .exp()?
            .neg()?
            .reshape([spec.heads, 1])?;
        let scan = |spans: SequenceLayout<'_>| {
            features.selective_scan(
                &step,
                &step,
                &rates,
                &written,
                &read,
                cache.recurrent(),
                spans,
            )
        };
        let (scanned, left) = scan(spans)?;
        cache.write_step(spans, &convolution, &left, |first| {
            Ok((
                channels.causal_conv1d_state(cache.convolution(), first)?,
                scan(first)?.1,
            ))
        })?;
        let skip = state.tensor(&layers.skip)?.reshape([spec.heads, 1])?;
        let output = scanned
            .add(&features.mul(&skip)?)?
            .reshape_with(&[tokens.clone(), inner.into()])?;
        let gated = match &layers.norm {
            Some(norm) => {
                let gated = output
                    .cast(DataType::F32)?
                    .mul(&gate.cast(DataType::F32)?.silu()?)?
                    .reshape_with(&[
                        tokens.clone(),
                        norm.groups.into(),
                        (inner / norm.groups).into(),
                    ])?;
                gated
                    .mul(
                        &gated
                            .mul(&gated)?
                            .mean(&[-1], true)?
                            .add_scalar(f64::from(norm.eps))?
                            .rsqrt()?,
                    )?
                    .reshape_with(&[tokens, inner.into()])?
                    .cast(data_type)?
                    .mul(state.tensor(&norm.weight)?)?
            }
            None => output.mul(&gate.silu()?)?,
        };
        gated.apply(&layers.output, state)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RgLruSpec {
    pub width: i64,
    pub blocks: i64,
    pub kernel: i64,
}

impl RgLruSpec {
    #[must_use]
    pub const fn cache(&self) -> StateCacheSpec {
        StateCacheSpec {
            channels: self.width,
            width: self.kernel,
            heads: self.width,
            rows: 1,
            columns: 1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RgLruLayers {
    pub input: Linear,
    pub gate: Linear,
    pub convolution: StateKey,
    pub convolution_bias: Option<StateKey>,
    pub input_gate: [StateKey; 2],
    pub recurrence_gate: [StateKey; 2],
    pub rate: StateKey,
    pub output: Linear,
}

#[derive(Debug, Clone)]
pub struct RgLru {
    layers: RgLruLayers,
    spec: RgLruSpec,
}

impl RgLru {
    pub fn create(layers: RgLruLayers, spec: RgLruSpec) -> Result<Self> {
        if spec.width <= 0 || spec.blocks <= 0 || spec.width % spec.blocks != 0 || spec.kernel < 2 {
            return Err(Error::invalid_argument("rg-lru sizes"));
        }
        Ok(Self { layers, spec })
    }
}

impl DecodingLayer for RgLru {
    type Cache = StateCache;

    fn decode(&self, input: &AttentionInput<'_, StateCache>, state: &ModelState) -> Result<Tensor> {
        let (layers, spec) = (&self.layers, self.spec);
        let (hidden, cache, spans) = (input.hidden, input.cache, input.step.spans());
        let data_type = hidden.data_type();
        let tokens = hidden.dimension(0)?;
        let block = spec.width / spec.blocks;
        let branch = hidden.apply(&layers.gate, state)?.gelu()?;
        let channels = hidden.apply(&layers.input, state)?;
        let (convolved, convolution) = convolve(
            &channels,
            &layers.convolution,
            layers.convolution_bias.as_ref(),
            cache,
            spans,
            state,
        )?;
        let blocked = convolved
            .reshape_with(&[tokens.clone(), spec.blocks.into(), block.into()])?
            .transpose(0, 1)?;
        let gate = |[weight, bias]: &[StateKey; 2]| {
            blocked
                .matmul(state.tensor(weight)?)?
                .add(&state.tensor(bias)?.unsqueeze(1)?)?
                .transpose(0, 1)?
                .reshape_with(&[tokens.clone(), spec.width.into()])?
                .sigmoid()
        };
        let (input_gate, recurrence_gate) =
            (gate(&layers.input_gate)?, gate(&layers.recurrence_gate)?);
        let log_decay = recurrence_gate
            .mul_scalar(-8.0)?
            .mul(&state.tensor(&layers.rate)?.softplus(1.0)?)?;
        let decay = log_decay.exp()?;
        let zero = capture::literal(hidden.device(), |device| {
            Tensor::from_vec(device, Shape::create([1])?, vec![0_i64])
        })?;
        let fresh = input.step.positions().equal(&zero)?.unsqueeze(1)?;
        let one = capture::literal(hidden.device(), |device| {
            Tensor::from_vec(device, Shape::create([1, 1])?, vec![1.0_f32])
        })?;
        let normalizer = Tensor::where_select(
            &fresh,
            &one.cast(data_type)?,
            &log_decay.mul_scalar(2.0)?.exp()?.rsub_scalar(1.0)?.sqrt()?,
        )?;
        let normalized = convolved
            .mul(&input_gate)?
            .mul(&normalizer)?
            .reshape_with(&[tokens.clone(), spec.width.into(), 1.into()])?;
        let steps = decay.cast(DataType::F32)?.log()?;
        let unit = |shape: &[DimensionExpr], data_type: DataType| -> Result<Tensor> {
            one.cast(data_type)?
                .reshape(vec![1_i64; shape.len()])?
                .expand_with(shape)?
                .contiguous()
        };
        let strength = unit(&[tokens.clone(), spec.width.into()], DataType::F32)?;
        let rates = unit(&[spec.width.into(), 1.into()], DataType::F32)?;
        let projection = unit(&[tokens.clone(), 1.into(), 1.into()], data_type)?;
        let scan = |spans: SequenceLayout<'_>| {
            normalized.selective_scan(
                &steps,
                &strength,
                &rates,
                &projection,
                &projection,
                cache.recurrent(),
                spans,
            )
        };
        let (scanned, left) = scan(spans)?;
        cache.write_step(spans, &convolution, &left, |first| {
            Ok((
                channels.causal_conv1d_state(cache.convolution(), first)?,
                scan(first)?.1,
            ))
        })?;
        scanned
            .reshape_with(&[tokens, spec.width.into()])?
            .mul(&branch)?
            .apply(&layers.output, state)
    }
}
