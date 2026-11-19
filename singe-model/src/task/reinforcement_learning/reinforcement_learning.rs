// `reinforcement-learning`: a trajectory's states, actions, returns to go, and timesteps to each step's predicted action, next state, and next return, as `transformers`' `DecisionTransformerModel` returns them; `transformers` has no pipeline for it.

use crate::{
    error::Result,
    head::{Trajectory, TrajectoryModel, TrajectoryPredictions},
    model,
    source::{LoadOptions, Source},
    task::Pipeline,
};

#[derive(Debug)]
pub struct ReinforcementLearningPipeline {
    model: Box<dyn TrajectoryModel>,
}

impl Pipeline for ReinforcementLearningPipeline {
    type Input = Trajectory;

    type Parameters = ();

    type Output = TrajectoryPredictions;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            model: model::load(source, options)?,
        })
    }

    fn run_one(&self, trajectory: &Self::Input, (): &Self::Parameters) -> Result<Self::Output> {
        self.model.predict(trajectory)
    }
}
