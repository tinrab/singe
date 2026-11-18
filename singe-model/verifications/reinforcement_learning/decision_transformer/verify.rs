// The Decision Transformer, predicting each step of a trajectory, as `verify.py` predicts them with `transformers`.

use singe::DataType;
use singe_model::task::reinforcement_learning::reinforcement_learning::ReinforcementLearningPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<ReinforcementLearningPipeline>("trajectory", DataType::F32);
}
