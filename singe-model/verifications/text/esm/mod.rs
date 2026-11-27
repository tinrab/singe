// ESM-2, filling masked residues of protein sequences, as `verify.py` fills them with `transformers`' fill-mask pipeline.

use singe_model::task::text::fill_mask::FillMaskPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<FillMaskPipeline>("fill");
    session.check::<FillMaskPipeline>("fill-two");
}
