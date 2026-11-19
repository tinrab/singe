use crate::{
    data_type::DataType,
    error::Result,
    nn::{Activation, Layer, Linear, ModelState, StateKey, layers::linear::joined_forward},
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

    #[must_use]
    pub const fn siblings(&self) -> [&Linear; 2] {
        [&self.gate, &self.up]
    }
}

impl Layer for GatedMlp {
    fn forward(&self, input: &Tensor, state: &ModelState) -> Result<Tensor> {
        let [gate, up] = if let Some(joined) = joined_forward(&self.siblings(), input, state)? {
            joined
        } else {
            [
                input.apply(&self.gate, state)?,
                input.apply(&self.up, state)?,
            ]
        };
        // SiLU's gate and the product run as one kernel.
        let gated = if matches!(self.activation, Activation::Silu) {
            up.swiglu(&gate)?
        } else {
            gate.apply(&self.activation, state)?.mul(&up)?
        };
        gated.apply(&self.down, state)
    }
}

// How a mixture-of-experts layer chooses experts for a token.
#[derive(Debug, Clone)]
pub enum MoeRouting {
    Grouped {
        bias: StateKey,
        groups: i64,
        kept_groups: i64,
        top_k: i64,
        scale: f64,
    },
    Softmax {
        top_k: i64,
    },
}

// The state a mixture-of-experts layer reads.
#[derive(Debug, Clone)]
pub struct MoeLayers {
    // The router's weight `[E, H]`.
    pub router: StateKey,
    // Every expert's gate and up projections, `[E, 2I, H]`, gate first, and down projection, `[E, H, I]`.
    pub gate_up: StateKey,
    pub down: StateKey,
    // What every token also passes through, scaled by the sigmoid of `shared_gate`, `[H]` to one, if given.
    pub shared: GatedMlp,
    pub shared_gate: Option<Box<Linear>>,
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
            gate_up,
            down,
            shared,
            shared_gate,
        } = &self.layers;
        let router = state.tensor(router)?.transpose(0, 1)?;
        let (experts, weights) = match &self.routing {
            // DeepSeek's routing runs in `F32` whatever the model's type, as the references do.
            MoeRouting::Grouped {
                bias,
                groups,
                kept_groups,
                top_k,
                scale,
            } => input
                .cast(DataType::F32)?
                .matmul(&router.cast(DataType::F32)?)?
                .grouped_topk_routing(state.tensor(bias)?, *groups, *kept_groups, *top_k, *scale)?,
            MoeRouting::Softmax { top_k } => {
                let (experts, weights) = input.matmul(&router)?.softmax_topk_routing(*top_k)?;
                (experts, weights.cast(input.data_type())?)
            }
        };
        let mut shared_output = input.apply(shared, state)?;
        if let Some(gate) = shared_gate {
            shared_output = shared_output.mul(&input.apply(gate.as_ref(), state)?.sigmoid()?)?;
        }

        input
            .moe(
                &experts,
                &weights,
                state.tensor(gate_up)?,
                state.tensor(down)?,
            )?
            .add(&shared_output)
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
