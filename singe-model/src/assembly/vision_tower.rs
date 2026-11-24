//! What a vision transformer's tower shares: its patches' embedding and the config of a tower of pre-norm blocks over patches of a square image, as `ViT`, `DeiT`, CLIP's and `SmolVLM`'s vision towers have it.

use serde::{Deserialize, Serialize};
use singe::{
    Result as SingeResult,
    nn::{Conv2dConfig, PatchEmbedding},
};

use crate::{assembly::PreNormEncoderConfig, schema::Schema};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VisionTowerConfig {
    pub encoder: PreNormEncoderConfig,
    pub layers: i64,
    /// The square image side the positions are learned for, a whole number of patches.
    pub image_size: i64,
    pub patch_size: i64,
    pub channels: i64,
}

impl VisionTowerConfig {
    #[must_use]
    pub const fn side(&self) -> i64 {
        self.image_size / self.patch_size
    }

    #[must_use]
    pub const fn patches(&self) -> i64 {
        self.side() * self.side()
    }
}

pub fn patch_embedding(
    schema: &mut Schema<'_>,
    channels: i64,
    width: i64,
    patch: i64,
    bias: bool,
) -> SingeResult<PatchEmbedding> {
    let patches = schema.conv2d(
        "patches",
        channels,
        width,
        patch,
        Conv2dConfig {
            stride: (patch, patch),
            ..Conv2dConfig::default()
        },
        bias,
    )?;
    Ok(PatchEmbedding::new(patches))
}
