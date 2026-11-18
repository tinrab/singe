// Verifications of the models under `model::vision`.

#[path = "depth_anything/verify.rs"]
mod depth_anything;
#[path = "detr/verify.rs"]
mod detr;
#[path = "dinov2/verify.rs"]
mod dinov2;
#[path = "lightglue/verify.rs"]
mod lightglue;
#[path = "sam/verify.rs"]
mod sam;
#[path = "segformer/verify.rs"]
mod segformer;
#[path = "superpoint/verify.rs"]
mod superpoint;
#[path = "swin2sr/verify.rs"]
mod swin2sr;
#[path = "vit/verify.rs"]
mod vit;
#[path = "yolos/verify.rs"]
mod yolos;

use crate::support::Verification;

pub const VERIFICATIONS: &[Verification] = &[
    Verification {
        name: "vision/depth_anything",
        verify: depth_anything::verify,
    },
    Verification {
        name: "vision/detr",
        verify: detr::verify,
    },
    Verification {
        name: "vision/dinov2",
        verify: dinov2::verify,
    },
    Verification {
        name: "vision/lightglue",
        verify: lightglue::verify,
    },
    Verification {
        name: "vision/sam",
        verify: sam::verify,
    },
    Verification {
        name: "vision/segformer",
        verify: segformer::verify,
    },
    Verification {
        name: "vision/superpoint",
        verify: superpoint::verify,
    },
    Verification {
        name: "vision/swin2sr",
        verify: swin2sr::verify,
    },
    Verification {
        name: "vision/vit",
        verify: vit::verify,
    },
    Verification {
        name: "vision/yolos",
        verify: yolos::verify,
    },
];
