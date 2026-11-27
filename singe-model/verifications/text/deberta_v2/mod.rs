// DeBERTa-v3, an inference model classifying zero-shot and text pairs, as `verify.py` classifies them with `transformers`' pipelines.

use singe_model::task::text::{
    text_classification::TextClassificationPipeline,
    zero_shot_classification::ZeroShotClassificationPipeline,
};

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<ZeroShotClassificationPipeline>("zero-shot");
    session.check::<TextClassificationPipeline>("pairs");
    session.check::<TextClassificationPipeline>("long-pair");
}
