// RoBERTa, through the tasks its snapshots here serve, as `verify.py` runs them in `transformers`: filling masks, among targets or not, and an inference model classifying zero-shot and text pairs.

use singe::DataType;
use singe_model::task::text::{
    fill_mask::FillMaskPipeline, text_classification::TextClassificationPipeline,
    zero_shot_classification::ZeroShotClassificationPipeline,
};

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<FillMaskPipeline>("fill", DataType::F32);
    session.check::<FillMaskPipeline>("fill-targets", DataType::F32);
    session.check::<ZeroShotClassificationPipeline>("zero-shot", DataType::F32);
    session.check::<TextClassificationPipeline>("pairs", DataType::F32);
}
