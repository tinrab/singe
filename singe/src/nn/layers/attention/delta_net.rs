use crate::{
    data_type::DataType,
    error::{Error, Result},
    nn::{
        AttentionInput, DecodingLayer, Layer, Linear, ModelState, RmsNorm, StateCache,
        StateCacheSpec, StateKey,
    },
    tensor::Tensor,
};

// The sizes of a [`GatedDeltaNet`]: `key_heads` heads of `key_dim` for queries and keys, `value_heads` heads of `value_dim` for values, and a convolution of `kernel` taps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GatedDeltaNetSpec {
    pub key_heads: i64,
    pub key_dim: i64,
    pub value_heads: i64,
    pub value_dim: i64,
    pub kernel: i64,
}

impl GatedDeltaNetSpec {
    // The convolution's channels: queries, keys, and values side by side.
    #[must_use]
    pub const fn channels(&self) -> i64 {
        2 * self.key_heads * self.key_dim + self.value_heads * self.value_dim
    }

    #[must_use]
    pub const fn cache(&self) -> StateCacheSpec {
        StateCacheSpec {
            channels: self.channels(),
            width: self.kernel,
            heads: self.value_heads,
            key_dim: self.key_dim,
            value_dim: self.value_dim,
        }
    }
}

// The layers and parameters of a [`GatedDeltaNet`], by their roles.
#[derive(Debug, Clone)]
pub struct GatedDeltaNetLayers {
    // Projects to queries, keys, and values side by side.
    pub qkv: Linear,
    // Projects to the gate the output norm multiplies by through SiLU.
    pub gate: Linear,
    // Projects to each value head's write strength, through a sigmoid.
    pub strength: Linear,
    // Projects to each value head's decay, before its bias and softplus.
    pub decay: Linear,
    // The depthwise convolution's taps, `[C, 1, K]`.
    pub convolution: StateKey,
    // Each value head's decay rate, as its logarithm, `[Hv]`.
    pub decay_log: StateKey,
    // Each value head's decay bias, `[Hv]`.
    pub decay_bias: StateKey,
    // The norm over each value head's output, which the gate then scales.
    pub norm: RmsNorm,
    pub output: Linear,
}

// Linear attention through the gated delta rule, as Qwen3-Next and Qwen3.5 write it, reading and extending a [`StateCache`].
//
// Queries, keys, and values pass through a depthwise causal convolution and SiLU; queries and keys are L2-normalized in `F32`, and queries scaled by one over the root of their dimension.
// The rule decays each value head's state by `exp(g)`, where `g = −exp(A_log) · softplus(a + dt_bias)`, and writes `sigmoid(b) · (v − read)` at the key.
#[derive(Debug, Clone)]
pub struct GatedDeltaNet {
    layers: GatedDeltaNetLayers,
    spec: GatedDeltaNetSpec,
}

// The epsilon of the queries' and keys' L2 norms, as the reference kernels add it.
const UNIT_EPS: f64 = 1e-6;

impl GatedDeltaNet {
    // Stores `layers`, whose value heads must be a multiple of the key heads.
    pub fn create(layers: GatedDeltaNetLayers, spec: GatedDeltaNetSpec) -> Result<Self> {
        if spec.key_heads <= 0
            || spec.value_heads % spec.key_heads != 0
            || spec.key_dim <= 0
            || spec.value_dim <= 0
            || spec.kernel < 2
        {
            return Err(Error::invalid_argument("gated delta net heads"));
        }
        Ok(Self { layers, spec })
    }

    #[must_use]
    pub const fn spec(&self) -> GatedDeltaNetSpec {
        self.spec
    }
}

// Divides each head of `heads` by its L2 norm, in `F32`.
fn unit(heads: &Tensor) -> Result<Tensor> {
    let heads = heads.cast(DataType::F32)?;
    heads.mul(
        &heads
            .mul(&heads)?
            .sum(&[-1], true)?
            .add_scalar(UNIT_EPS)?
            .rsqrt()?,
    )
}

impl DecodingLayer for GatedDeltaNet {
    type Cache = StateCache;

    #[expect(
        clippy::cast_precision_loss,
        reason = "a head dimension is far below where `F64` rounds it"
    )]
    fn decode(&self, input: &AttentionInput<'_, StateCache>, state: &ModelState) -> Result<Tensor> {
        let (layers, spec) = (&self.layers, self.spec);
        let (hidden, cache, spans) = (input.hidden, input.cache, input.step.spans());
        let data_type = hidden.data_type();
        let tokens = hidden.dimension(0)?;
        let mixed = hidden.apply(&layers.qkv, state)?;
        let taps = state
            .tensor(&layers.convolution)?
            .reshape([spec.channels(), spec.kernel])?;
        let convolved = mixed
            .causal_conv1d(&taps, cache.convolution(), spans)?
            .silu()?;
        let convolution = mixed.causal_conv1d_state(cache.convolution(), spans)?;
        let heads = |start: i64, heads: i64, dim: i64| {
            convolved
                .narrow(-1, start, heads * dim)?
                .contiguous()?
                .reshape_with(&[tokens.clone(), heads.into(), dim.into()])
        };
        let key_width = spec.key_heads * spec.key_dim;
        let repeats = spec.value_heads / spec.key_heads;
        let repeated = |heads: Tensor| {
            if repeats > 1 {
                heads.repeat_interleave(repeats, 1)
            } else {
                Ok(heads)
            }
        };
        let query = repeated(
            unit(&heads(0, spec.key_heads, spec.key_dim)?)?
                .mul_scalar((spec.key_dim as f64).sqrt().recip())?,
        )?;
        let key = repeated(unit(&heads(key_width, spec.key_heads, spec.key_dim)?)?)?;
        let value = heads(2 * key_width, spec.value_heads, spec.value_dim)?.cast(DataType::F32)?;
        let strength = hidden
            .apply(&layers.strength, state)?
            .sigmoid()?
            .cast(DataType::F32)?;
        let rate = state
            .tensor(&layers.decay_log)?
            .cast(DataType::F32)?
            .exp()?
            .neg()?;
        let decay = hidden
            .apply(&layers.decay, state)?
            .cast(DataType::F32)?
            .add(&state.tensor(&layers.decay_bias)?.cast(DataType::F32)?)?
            .softplus(1.0)?
            .mul(&rate)?;
        let recurrent = cache.recurrent();
        let attended = query.gated_delta_rule(&key, &value, &decay, &strength, recurrent, spans)?;
        let recurrent =
            query.gated_delta_rule_state(&key, &value, &decay, &strength, recurrent, spans)?;
        cache.write(&convolution, &recurrent)?;
        let head_shape = [
            tokens.clone(),
            spec.value_heads.into(),
            spec.value_dim.into(),
        ];
        let gate = hidden
            .apply(&layers.gate, state)?
            .reshape_with(&head_shape)?
            .cast(DataType::F32)?
            .silu()?;
        let normed = layers
            .norm
            .forward(&attended.cast(data_type)?, state)?
            .cast(DataType::F32)?
            .mul(&gate)?
            .cast(data_type)?;
        normed
            .reshape_with(&[tokens, (spec.value_heads * spec.value_dim).into()])?
            .apply(&layers.output, state)
    }
}
