//! Models `transformers` documents among its vision models.

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

use crate::model::ModelEntry;

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
