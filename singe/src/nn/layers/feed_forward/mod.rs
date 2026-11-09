use crate::{
    data_type::DataType,
    error::Result,
    nn::{Activation, Layer, Linear, ModelState, StateKey},
    tensor::Tensor,
};

// A feed-forward layer: `down(activation(up(x)))`.
#[derive(Debug, Clone)]
pub struct Mlp {
    up: Linear,
    down: Linear,
    activation: Activation,
}

impl Mlp {
    #[must_use]
    pub const fn new(up: Linear, down: Linear, activation: Activation) -> Self {
        Self {
            up,
            down,
            activation,
        }
    }
}

impl Layer for Mlp {
    fn forward(&self, input: &Tensor, state: &ModelState) -> Result<Tensor> {
        input
            .apply(&self.up, state)?
            .apply(&self.activation, state)?
            .apply(&self.down, state)
    }
}

// A gated feed-forward layer: `down(activation(gate(x)) * up(x))`.
#[derive(Debug, Clone)]
pub struct GatedMlp {
    gate: Linear,
    up: Linear,
    down: Linear,
    activation: Activation,
}

impl GatedMlp {
    // Stores the three projections and the gate's activation.
    #[must_use]
    pub const fn new(gate: Linear, up: Linear, down: Linear, activation: Activation) -> Self {
        Self {
            gate,
            up,
            down,
            activation,
        }
    }
}

impl Layer for GatedMlp {
    fn forward(&self, input: &Tensor, state: &ModelState) -> Result<Tensor> {
        input
            .apply(&self.gate, state)?
            .apply(&self.activation, state)?
            .mul(&input.apply(&self.up, state)?)?
            .apply(&self.down, state)
    }
}

// How a mixture-of-experts layer chooses experts for a token, as `Tensor::grouped_topk_routing` takes it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoeRouting {
    pub groups: i64,
    pub kept_groups: i64,
    pub top_k: i64,
    pub scale: f64,
}

// The state a mixture-of-experts layer reads.
#[derive(Debug, Clone)]
pub struct MoeLayers {
    // The router's weight `[E, H]`, and what it adds to each expert's score when choosing.
    pub router: StateKey,
    pub router_bias: StateKey,
    // Every expert's gate and up projections, `[E, 2I, H]`, gate first, and down projection, `[E, H, I]`.
    pub gate_up: StateKey,
    pub down: StateKey,
    // What every token also passes through.
    pub shared: GatedMlp,
}

// A mixture-of-experts layer over `[T, H]` tokens, whose experts' gates are SiLU.
// Each token goes through the experts a router chooses for it, weighted, and through a shared expert.
#[derive(Debug, Clone)]
pub struct MixtureOfExperts {
    layers: MoeLayers,
    routing: MoeRouting,
}

impl MixtureOfExperts {
    #[must_use]
    pub const fn new(layers: MoeLayers, routing: MoeRouting) -> Self {
        Self { layers, routing }
    }
}

impl Layer for MixtureOfExperts {
    fn forward(&self, input: &Tensor, state: &ModelState) -> Result<Tensor> {
        let MoeLayers {
            router,
            router_bias,
            gate_up,
            down,
            shared,
        } = &self.layers;
        let MoeRouting {
            groups,
            kept_groups,
            top_k,
            scale,
        } = self.routing;
        // Routing runs in `F32` whatever the model's type, as the references do.
        let logits = input
            .cast(DataType::F32)?
            .matmul(&state.tensor(router)?.cast(DataType::F32)?.transpose(0, 1)?)?;
        let (experts, weights) = logits.grouped_topk_routing(
            state.tensor(router_bias)?,
            groups,
            kept_groups,
            top_k,
            scale,
        )?;
        input
            .moe(
                &experts,
                &weights,
                state.tensor(gate_up)?,
                state.tensor(down)?,
            )?
            .add(&input.apply(shared, state)?)
    }
}

// The feed-forward layer of a decoder whose layers differ, dense in some and MoE in others.
#[derive(Debug, Clone)]
pub enum DecoderFeedForward {
    Dense(GatedMlp),
    Moe(MixtureOfExperts),
}

impl Layer for DecoderFeedForward {
    fn forward(&self, input: &Tensor, state: &ModelState) -> Result<Tensor> {
        match self {
            Self::Dense(layer) => layer.forward(input, state),
            Self::Moe(layer) => layer.forward(input, state),
        }
    }
}
