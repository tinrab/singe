// Shap-E's image variant, generating a mesh from an image and a seed, as `verify.py` generates it with `diffusers`.

use singe_model::task::vision::image_to_3d::ImageTo3dPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<ImageTo3dPipeline>("mesh");
}
