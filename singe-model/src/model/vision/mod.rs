//! Models `transformers` documents among its vision models.

pub mod convnext;
pub mod deformable_detr;
pub mod depth_anything;
pub mod detr;
pub mod dinov2;
pub mod efficientloftr;
pub mod efficientnet;
pub mod lightglue;
pub mod mask2former;
pub mod mobilenet_v2;
pub mod rt_detr;
pub mod sam;
pub mod sam2;
pub mod sam3;
pub mod segformer;
pub mod superglue;
pub mod superpoint;
pub mod swin2sr;
pub mod upernet;
pub mod vit;
pub mod yolos;

use crate::model::ModelEntry;

pub(crate) const MODELS: &[&ModelEntry] = &[
    &convnext::MODEL,
    &deformable_detr::MODEL,
    &depth_anything::MODEL,
    &detr::MODEL,
    &dinov2::MODEL,
    &efficientloftr::MODEL,
    &efficientnet::MODEL,
    &lightglue::MODEL,
    &mask2former::MODEL,
    &mobilenet_v2::MODEL,
    &rt_detr::MODEL,
    &sam::MODEL,
    &sam2::MODEL,
    &sam3::MODEL,
    &segformer::MODEL,
    &superglue::MODEL,
    &superpoint::MODEL,
    &swin2sr::MODEL,
    &upernet::MODEL,
    &vit::MODEL,
    &yolos::MODEL,
];
