//! Models that `diffusers` defines: its autoencoders, its U-Nets, and Shap-E's prior and renderer.

pub mod autoencoder_kl;
pub mod autoencoder_tiny;
pub mod shap_e;
pub mod unet_2d;
pub mod unet_2d_condition;

use crate::model::ModelEntry;

pub(crate) const MODELS: &[&ModelEntry] = &[
    &autoencoder_kl::MODEL,
    &autoencoder_tiny::MODEL,
    &unet_2d::MODEL,
    &unet_2d_condition::MODEL,
];
