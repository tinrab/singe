// Linear layers with ReLU between them, as SAM's `SamFeedForward` and DETR's and YOLOS's `*MLPPredictionHead` are, each layer named by its index.

use singe::{
    Result as SingeResult, Tensor,
    nn::{Activation, Layer, Linear, ModelState},
};

use crate::schema::Schema;

#[derive(Debug, Clone)]
pub struct Mlp(Vec<Linear>);

impl Mlp {
    // Registers `depth` layers, `0` to `depth - 1`: the first from `input` features, the last to `out`, and `hidden` between them.
    pub fn create(
        schema: &mut Schema<'_>,
        input: i64,
        hidden: i64,
        out: i64,
        depth: i64,
    ) -> SingeResult<Self> {
        (0..depth)
            .map(|index| {
                let from = if index == 0 { input } else { hidden };
                let to = if index + 1 == depth { out } else { hidden };
                schema.linear(&index.to_string(), from, to, true)
            })
            .collect::<SingeResult<Vec<_>>>()
            .map(Self)
    }
}

impl Layer for Mlp {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let mut hidden = input.clone();
        for (index, layer) in self.0.iter().enumerate() {
            hidden = hidden.apply(layer, state)?;
            if index + 1 < self.0.len() {
                hidden = hidden.apply(&Activation::Relu, state)?;
            }
        }
        Ok(hidden)
    }
}
