// Linear layers with an activation between them, as SAM's `SamFeedForward`, DETR's and YOLOS's `*MLPPredictionHead`, and OWL-ViT's box head are, each layer named by its index.

use singe::{
    Result as SingeResult, Tensor,
    nn::{Activation, Layer, Linear, ModelState},
};

use crate::schema::Schema;

#[derive(Debug, Clone)]
pub struct LinearStack {
    layers: Vec<Linear>,
    activation: Activation,
}

impl LinearStack {
    // Registers `depth` layers, `0` to `depth - 1`: the first from `input` features, the last to `output`, and `hidden` between them.
    pub fn create(
        schema: &mut Schema<'_>,
        [input, hidden, output]: [i64; 3],
        depth: i64,
        activation: Activation,
    ) -> SingeResult<Self> {
        let layers = (0..depth)
            .map(|index| {
                let from = if index == 0 { input } else { hidden };
                let to = if index + 1 == depth { output } else { hidden };
                schema.linear(&index.to_string(), from, to, true)
            })
            .collect::<SingeResult<Vec<_>>>()?;
        Ok(Self { layers, activation })
    }
}

impl Layer for LinearStack {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let mut hidden = input.clone();
        for (index, layer) in self.layers.iter().enumerate() {
            hidden = hidden.apply(layer, state)?;
            if index + 1 < self.layers.len() {
                hidden = hidden.apply(&self.activation, state)?;
            }
        }
        Ok(hidden)
    }
}
