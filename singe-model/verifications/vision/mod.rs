// Verifications of the models under `model::vision`.

mod depth_anything;
mod detr;
mod dinov2;
mod lightglue;
mod sam;
mod segformer;
mod superpoint;
mod swin2sr;
mod vit;
mod yolos;

use crate::support::verifications;

verifications!(
    "vision",
    [
        depth_anything,
        detr,
        dinov2,
        lightglue,
        sam,
        segformer,
        superpoint,
        swin2sr,
        vit,
        yolos,
    ]
);
