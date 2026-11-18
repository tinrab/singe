// Shap-E, generating a mesh from a prompt and a seed, as `verify.py` generates it with `diffusers`.

use singe::DataType;
use singe_model::task::vision::text_to_3d::TextTo3dPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<TextTo3dPipeline>("mesh", DataType::F32);
}
