//! The mixture of experts of a Llama-family decoder layer, and the rules that import its experts.
//!
//! Singe packs a layer's experts as `feed_forward.experts.gate_up`, `[E, 2 · I, W]`, each expert's gate rows then its up rows, and `feed_forward.experts.down`, `[E, W, I]`.

use serde::{Deserialize, Serialize};
use singe::{
    Result as SingeResult,
    nn::{Activation, MixtureOfExperts, MoeLayers, MoeRouting, StatePiece, TensorSource},
};

use crate::{
    format::{Matched, Rules, join},
    schema::Schema,
};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ExpertRouting {
    Softmax,
    Grouped {
        groups: i64,
        kept_groups: i64,
        scale: f64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ExpertsConfig {
    pub count: i64,
    /// The number of experts the router chooses for each token.
    pub top_k: i64,
    /// Each expert's inner width.
    pub inner: i64,
    /// The inner width of the shared expert every token also passes through.
    pub shared_inner: i64,
    pub routing: ExpertRouting,
}

pub fn mixture_of_experts(
    schema: &mut Schema<'_>,
    config: &ExpertsConfig,
    width: i64,
    activation: Activation,
) -> SingeResult<MixtureOfExperts> {
    let ExpertsConfig {
        count,
        top_k,
        inner,
        shared_inner,
        routing,
    } = *config;
    let mut feed_forward = schema.scope("feed_forward");
    let router = feed_forward.parameter("router.weight", [count, width])?;
    let gate_up = feed_forward.parameter("experts.gate_up", [count, 2 * inner, width])?;
    let down = feed_forward.parameter("experts.down", [count, width, inner])?;
    let shared =
        feed_forward.gated_mlp("shared", [width, shared_inner, width], activation, false)?;
    let (shared_gate, routing) = match routing {
        ExpertRouting::Softmax => (
            Some(Box::new(feed_forward.linear(
                "shared_gate",
                width,
                1,
                false,
            )?)),
            MoeRouting::Softmax { top_k },
        ),
        ExpertRouting::Grouped {
            groups,
            kept_groups,
            scale,
        } => (
            None,
            MoeRouting::Grouped {
                bias: feed_forward.parameter("router.bias", [count])?,
                groups,
                kept_groups,
                top_k,
                scale,
            },
        ),
    };
    Ok(MixtureOfExperts::new(
        MoeLayers {
            router,
            gate_up,
            down,
            shared,
            shared_gate,
        },
        routing,
    ))
}

/// Imports experts stored one by one (`mlp.experts.E.gate_proj`) or packed as `transformers` 5 holds them (`mlp.experts.gate_up_proj`), of a decoder under `model` whose experts are `inner` wide.
#[must_use]
pub fn expert_rules(model: &str, inner: i64) -> Rules {
    let piece = move |matched: &Matched<'_>, packed: &str, row: i64| {
        let expert = matched.number(1)?;
        Ok(vec![(
            matched.target(&format!("blocks.*.feed_forward.experts.{packed}")),
            TensorSource::Pieces(vec![StatePiece {
                name: matched.name.to_owned(),
                origin: vec![expert, row, 0],
            }]),
        )])
    };
    let experts = join(model, "layers.*.mlp.experts");
    Rules::new()
        .custom(&format!("{experts}.*.gate_proj.weight"), move |matched| {
            piece(matched, "gate_up", 0)
        })
        .custom(&format!("{experts}.*.up_proj.weight"), move |matched| {
            piece(matched, "gate_up", inner)
        })
        .custom(&format!("{experts}.*.down_proj.weight"), move |matched| {
            piece(matched, "down", 0)
        })
        .rename(
            &format!("{experts}.gate_up_proj"),
            "blocks.*.feed_forward.experts.gate_up",
        )
        .rename(
            &format!("{experts}.down_proj"),
            "blocks.*.feed_forward.experts.down",
        )
}
