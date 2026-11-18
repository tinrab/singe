// Models `transformers` documents among its vision models, and the small pieces several of them share.

pub mod depth_anything;
pub mod detr;
pub mod dinov2;
pub mod lightglue;
pub mod sam;
pub mod segformer;
pub mod superpoint;
pub mod swin2sr;
pub mod vit;
pub mod yolos;

use singe::nn::Conv2dConfig;

use super::ModelEntry;

pub(crate) const MODELS: &[&ModelEntry] = &[
    &depth_anything::MODEL,
    &detr::MODEL,
    &dinov2::MODEL,
    &lightglue::MODEL,
    &sam::MODEL,
    &segformer::MODEL,
    &superpoint::MODEL,
    &swin2sr::MODEL,
    &vit::MODEL,
    &yolos::MODEL,
];

// A square convolution of odd `kernel` padded to keep the map's size.
pub(crate) fn same(kernel: i64) -> Conv2dConfig {
    let padding = kernel / 2;
    Conv2dConfig {
        padding: (padding, padding, padding, padding),
        ..Conv2dConfig::default()
    }
}
