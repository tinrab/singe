use singe::{
    Result as SingeResult, Tensor,
    nn::{Layer, Model, ModelState},
};

// An image encoder's layers, from images `[B, C, H, W]` to each token's last hidden state, `[B, T, D]`, with the pooling its model gives one vector per image by.
pub trait PooledLayer: Layer {
    // Returns one vector per image, `[B, D]`, from the hidden states.
    fn pooled(&self, hidden: &Tensor) -> SingeResult<Tensor>;
}

#[derive(Debug)]
pub struct ImageEncoderLayers(pub Box<dyn PooledLayer>);

impl ImageEncoderLayers {
    pub fn pooled(&self, hidden: &Tensor) -> SingeResult<Tensor> {
        self.0.pooled(hidden)
    }
}

impl Layer for ImageEncoderLayers {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        self.0.forward(input, state)
    }
}

pub type ImageEncoder = Model<ImageEncoderLayers>;
