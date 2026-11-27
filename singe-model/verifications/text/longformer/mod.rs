// Longformer, filling masks, one far past its window from the start, as `verify.py` fills them with `transformers`' fill-mask pipeline.

use singe_model::task::text::fill_mask::FillMaskPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<FillMaskPipeline>("fill");
}
