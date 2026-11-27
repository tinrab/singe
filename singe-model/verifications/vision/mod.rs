// Verifications of the models under `model::vision`.

mod convnext;
mod deformable_detr;
mod depth_anything;
mod detr;
mod dinov2;
mod efficientloftr;
mod efficientnet;
mod lightglue;
mod mask2former;
mod mobilenet_v2;
mod rt_detr;
mod sam;
mod sam2;
mod sam3;
mod segformer;
mod superglue;
mod superpoint;
mod swin2sr;
mod upernet;
mod vit;
mod yolos;

use crate::support::verifications;

verifications!(
    "vision",
    [
        convnext,
        deformable_detr,
        depth_anything,
        detr,
        dinov2,
        efficientloftr,
        efficientnet,
        lightglue,
        mask2former,
        mobilenet_v2,
        rt_detr,
        sam,
        sam2,
        sam3,
        segformer,
        superglue,
        superpoint,
        swin2sr,
        upernet,
        vit,
        yolos,
    ]
);
