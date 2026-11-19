use singe::{
    Result as SingeResult, Tensor,
    nn::{Layer, Model},
};

use crate::head::Boxed;

// An image encoder's layers, from images `[B, C, H, W]` to each token's last hidden state, `[B, T, D]`, with the pooling its model gives one vector per image by.
pub trait PooledLayer: Layer {
    // Returns one vector per image, `[B, D]`, from the hidden states.
    fn pooled(&self, hidden: &Tensor) -> SingeResult<Tensor>;
}

pub type ImageEncoder = Model<Boxed<dyn PooledLayer>>;
