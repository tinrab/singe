// Rules that import the experts of a Llama-family mixture of experts.
//
// Singe packs a layer's experts as `feed_forward.experts.gate_up`, `[E, 2 · I, W]`, each expert's gate rows then its up rows, and `feed_forward.experts.down`, `[E, W, I]`.

use singe::nn::{StatePiece, TensorSource};

use crate::format::{Matched, Rules};

// Imports experts stored one by one (`mlp.experts.E.gate_proj`) or packed as `transformers` 5 holds them (`mlp.experts.gate_up_proj`), of a decoder under `model` whose experts are `inner` wide.
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
    let experts = format!("{model}.layers.*.mlp.experts");
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
