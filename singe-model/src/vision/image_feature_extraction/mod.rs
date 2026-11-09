// Models that turn an image into features to compare or build on, Hugging Face's `image-feature-extraction`, each under its `model_type`.

pub mod dinov2;

use std::path::Path;

use singe::{
    DataType, Device, Result as SingeResult, Tensor,
    nn::{Layer, Model, ModelState},
};

use crate::{Error, Result, config::Snapshot};

// The encoder a snapshot's `model_type` names, from images `[B, C, H, W]` to each token's final hidden state, `[B, T, D]`.
#[derive(Debug, Clone)]
pub enum ImageEncoderLayers {
    Dinov2(dinov2::Dinov2Layers),
}

impl ImageEncoderLayers {
    // Returns one vector per image, `[B, D]`, as the model's pooler gives it.
    pub fn pooled(&self, hidden: &Tensor) -> SingeResult<Tensor> {
        match self {
            Self::Dinov2(_) => dinov2::class_embeddings(hidden),
        }
    }
}

impl Layer for ImageEncoderLayers {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        match self {
            Self::Dinov2(layers) => layers.forward(input, state),
        }
    }
}

pub type ImageEncoder = Model<ImageEncoderLayers>;

// Loads the encoder a snapshot's `model_type` names.
pub fn load_image_encoder(
    snapshot: impl AsRef<Path>,
    device: Device,
    data_type: DataType,
) -> Result<ImageEncoder> {
    let snapshot = snapshot.as_ref();
    match Snapshot::new(snapshot).model_type()?.as_str() {
        "dinov2" => {
            let model = dinov2::load(snapshot, device, data_type)?;
            Ok(Model::new(
                ImageEncoderLayers::Dinov2(model.layer().clone()),
                model.state().clone(),
            ))
        }
        other => Err(Error::UnsupportedConfig(format!("model type {other}"))),
    }
}
