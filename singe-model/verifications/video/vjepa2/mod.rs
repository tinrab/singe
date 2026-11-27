// V-JEPA 2, classifying a clip from its sampled frames, and pooling a base checkpoint's encoder states over a clip, as `verify.py` classifies and extracts them with `transformers`' processor and models.

use singe_model::task::vision::{
    video_classification::VideoClassificationPipeline,
    video_feature_extraction::VideoFeatureExtractionPipeline,
};

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<VideoClassificationPipeline>("classification");
    session.check::<VideoFeatureExtractionPipeline>("features");
}
