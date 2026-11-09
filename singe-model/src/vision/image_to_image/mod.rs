// Models that turn images into images, Hugging Face's `image-to-image`, each under the name of its architecture.
// An autoencoder is here, since encoding an image and decoding it again is the image-to-image use its pipelines make of it.

pub mod taesd;

use std::path::Path;

use singe::{
    DataType, Device, Result as SingeResult, Tensor,
    nn::{Model, Sequential},
};

use crate::{Error, Result, config::Snapshot};

// An image autoencoder's two halves: an encoder from images, `[B, C, H, W]` in [-1, 1], to latents, and a decoder from latents back to images.
// Each half has its own state, so each captures, tunes, and exports alone.
#[derive(Debug, Clone)]
pub struct Autoencoder {
    encoder: Model<Sequential>,
    decoder: Model<Sequential>,
}

impl Autoencoder {
    #[must_use]
    pub const fn new(encoder: Model<Sequential>, decoder: Model<Sequential>) -> Self {
        Self { encoder, decoder }
    }

    #[must_use]
    pub const fn encoder(&self) -> &Model<Sequential> {
        &self.encoder
    }

    #[must_use]
    pub const fn decoder(&self) -> &Model<Sequential> {
        &self.decoder
    }

    // Returns the latents `diffusers`' `encode` returns, before any pipeline's scaling factor.
    pub fn encode(&self, images: &Tensor) -> SingeResult<Tensor> {
        self.encoder.forward(images)
    }

    pub fn decode(&self, latents: &Tensor) -> SingeResult<Tensor> {
        self.decoder.forward(latents)
    }
}

// Loads the autoencoder a `diffusers` snapshot's `_class_name` names.
pub fn load_autoencoder(
    snapshot: impl AsRef<Path>,
    device: Device,
    data_type: DataType,
) -> Result<Autoencoder> {
    let snapshot = snapshot.as_ref();
    match Snapshot::new(snapshot).class_name()?.as_str() {
        "AutoencoderTiny" => taesd::load(snapshot, device, data_type),
        other => Err(Error::UnsupportedConfig(format!("diffusers class {other}"))),
    }
}
