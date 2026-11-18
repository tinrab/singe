// VideoMAE, classifying a clip from its sampled frames, as `verify.py` classifies them with `transformers`' processor and model.

use singe::DataType;
use singe_model::task::vision::video_classification::VideoClassificationPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<VideoClassificationPipeline>("classification", DataType::F32);
}
