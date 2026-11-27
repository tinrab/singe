// BigBird, filling masks, as `verify.py` fills them with `transformers`' fill-mask pipeline, under the length where both attend in full and past it, where both attend block-sparsely.

use singe_model::task::text::fill_mask::FillMaskPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<FillMaskPipeline>("fill");
    session.check::<FillMaskPipeline>("long");
}
