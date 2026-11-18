// XLM-RoBERTa, filling masks in two languages, as `verify.py` fills them with `transformers`' fill-mask pipeline.

use singe::DataType;
use singe_model::task::text::fill_mask::FillMaskPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<FillMaskPipeline>("fill", DataType::F32);
}
